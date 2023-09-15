# rewrite flash2 with alibi support, from scratch and first principles.
# 

# step 1 - get forward kernel going with online softmax.

import torch

from triton import cdiv, jit
from triton import language as tl


_supported_head_dims = (16,32,64,128)

@jit
def _fwd_kernel(
    q_in: torch.tensor, 
    k_in: torch.tensor, 
    v_in: torch.tensor,
    output_in: torch.tensor,
    qk_scale_factor: torch.float,
    softmax_normalizer: torch.tensor, 
    use_causal: tl.constexpr,
    use_mask: tl.constexpr, #: bool, 
    mask_in: torch.tensor,

    # strides
    q_stride_z,
    q_stride_h,
    q_stride_sq,
    q_stride_hd,
    k_stride_z,
    k_stride_h,
    k_stride_sq,
    k_stride_hd,
    v_stride_z,
    v_stride_h,
    v_stride_sq,
    v_stride_hd,
    mask_stride_z,
    mask_stride_h,
    mask_stride_sq,
    mask_stride_hd,
    o_stride_z,
    o_stride_h,
    o_stride_sq,
    o_stride_hd,

    num_heads: int,
    seq_len: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_head_dim: tl.constexpr,
    
    ):
    start_m = tl.program_id(0) # row offset
    offset_heads = tl.program_id(1) # heads offset

    qkv_offset = offset_heads * q_stride_h

    # create block pointers
    q_bpr = tl.make_block_ptr(
        base = q_in + qkv_offset,
        shape = (seq_len, block_head_dim),
        strides = (q_stride_sq, q_stride_hd),
        offsets = (start_m * block_m, 0),
        block_shape = (block_m, block_head_dim),
        order=(1,0),

    )

    k_bpr = tl.make_block_ptr(
        base = k_in + qkv_offset,
        shape = (block_head_dim, seq_len),
        strides = (k_stride_hd, k_stride_sq),
        offsets = (0,0),
        block_shape=(block_head_dim, block_n),
        order=(0,1),

    )

    v_bpr = tl.make_block_ptr(
        base = v_in + qkv_offset,
        shape = (seq_len, block_head_dim),
        strides = (v_stride_sq, v_stride_hd),
        offsets = (0,0),
        block_shape=(block_n, block_head_dim),
        order=(1,0),
    )

    if use_mask:
        mask_offset_z = offset_heads // num_heads
        mask_offset_heads = offset_heads % num_heads
        mask_bpr = tl.make_block_ptr(
            base = mask_in + mask_offset_z * mask_stride_z + mask_offset_heads * mask_stride_h,
            shape = (seq_len, seq_len),
            strides = (mask_stride_sq, mask_stride_hd),
            offsets=(start_m * block_m, 0),
            block_shape=(block_m, block_n),
            order=(1,0),
        )

    offsets_m = start_m * block_m + tl.arange(0,block_m)
    offsets_n = tl.arange(0,block_n)

    # init online softmax meta stores
    max_i = tl.zeros([block_m], dtype=tl.float32)+float("-inf")
    normalizer_i = tl.zeros([block_m], dtype=tl.float32)
    accumulator = tl.zeros([block_m, block_head_dim], dtype=tl.float32)

    # credit to: Adam P. Goucher ((https://github.com/apgoucher))
    # scale sm_scale by 1/log_2(e) and use 2^x
    qk_scale = qk_scale_factor * 1.44269504

    # q will stay in sram
    q = tl.load(q_bpr)
    q = (q * qk_scale).to(k_in.dtype.element_ty)

    low = 0
    high = (start_m+1)* block_m if use_causal else seq_len

    for start_n in range(low, high, block_n):
        k = tl.load(k_bpr)
        v = tl.load(v_bpr)

        # attn matrix calculation
        qk = tl.zeros([block_m, block_n], dtype=tl.float32)
        if use_mask:
            mask = tl.load(mask_bpr)
            qk += mask.to(qk.dtype)
        if use_causal:
            qk = tl.where(offsets_m[:,None]>= (start_n + offsets_n[None,:]), qk, float("-inf"))
        qk += tl.dot(q, k, allow_tf32=True)
        

        # online softmax updates
        max_i_new = tl.maximum(max_i, tl.max(qk, 1))
        alpha = tl.math.exp2(max_i - max_i_new)
        probs = tl.math.exp2(qk-max_i_new[:,None])

        # scale and update accumulator
        accumulator *= alpha[:, None]
        accumulator += tl.dot(probs.to(v_in.dtype.element_ty), v, allow_tf32=True)

        normalizer_i = normalizer_i * alpha + tl.sum(probs,1)
        max_i = max_i_new

        # move pointers
        k_bpr = tl.advance(k_bpr, (0, block_n))
        v_bpr = tl.advance(v_bpr, (block_n, 0))
        if use_mask:
            mask_bpr = tl.advance(mask_bpr, (0,block_n))

    accumulator = accumulator / normalizer_i[:,None]
    normalizer_ptrs = softmax_normalizer + offset_heads * seq_len+ offsets_m
    tl.store(normalizer_ptrs, max_i + tl.math.log2(normalizer_i))

    output_bpr = tl.make_block_ptr(
        base = output_in + qkv_offset,
        shape = (seq_len, block_head_dim),
        strides=(o_stride_sq, o_stride_hd),
        offsets = (start_m * block_m, 0),
        block_shape=(block_m, block_head_dim),
        order=(1,0),
    )
    tl.store(output_bpr, accumulator.to(k_in.dtype.element_ty))

@jit
def _bwd_preprocess(output_in,
                    do_in,
                    delta_in,
                    block_m: tl.constexpr,
                    dim_head: tl.constexpr,
                    ):
    offsets_m = tl.program_id(0) * block_m + tl.arange(0,block_m)
    offsets_n = tl.arange(0,dim_head)

    outs = tl.load(output_in + offsets_m[:,None] * dim_head + offsets_n[None,:])
    outs.to(tl.float32)

    do = tl.load(do_in + offsets_m[:,None] * dim_head + offsets_n[None,:])
    do.to(tl.float32)

    delta = tl.sum(outs * do, axis=1)
    tl.store(delta_in + offsets_m, delta)
    
@jit
def _bwd_kernel( q, k, v,
            qk_scaling,  # sm_scale
            mask,
            output, do,
            dq, dk, dv,
            softmax_normalizer, # L
            delta, #D 
            stride_dqa, # d of q alpha 
            q_stride_z, q_stride_h, 
            q_stride_sqlen, q_stride_hdim,
            k_stride_z, k_stride_h,
            k_stride_sqlen, k_stride_hdim,
            v_stride_z, v_stride_h,
            v_stride_sqlen, v_stride_hdim,
            mask_stride_z, mask_stride_h,
            mask_stride_sqlen, mask_stride_hdim,




            ):
    pass
    

class _newattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, qk_scaling=None, use_causal=False, use_mask = False, mask_in: torch.Tensor = None):
        qdim, kdim, vdim = q.shape[-1], k.shape[-1], v.shape[-1]
        print(f"{qdim=}")
        # confirm suitable qkv shapes
        assert qdim == kdim and kdim == vdim
        assert kdim in _supported_head_dims
        assert 4 == q.dim()
        # currently support only mask or only causal (mask should include causal)
        '''if use_causal:
            assert use_causal != use_mask, f"causal {use_causal=} and {use_mask=} are mutually exclusive"
        elif use_mask:
            assert use_mask != use_causal, f"using casual and mask together is not yet supported"
            assert mask_in is not None, f" use_mask set but no mask supplied in mask_in param"
        '''
        # block tuning
        block_m = 64 # 128 
        block_n = 32 
        print(f"block sizes: {block_m=}, {block_n=}")

        output = torch.empty_like(q)
        if qk_scaling is None:
            qk_scale_factor = kdim**0.5  
        else:
            qk_scale_factor = qk_scaling
        print(f"{qk_scale_factor=}")

        # triton tuning
        num_warps = 4 if kdim <= 64 else 8
        num_stages = 4

        softmax_normalizer = torch.empty((q.shape[0]*q.shape[1], q.shape[2]),device=q.device, dtype=torch.float32)
        
        print(f"{softmax_normalizer.shape=}")
        grid = (cdiv(q.shape[2], block_m ), q.shape[0] * q.shape[1],1)
        print(f"{grid=}")
        num_heads, seq_len = q.shape[1], q.shape[2]
        print(f"{num_heads=}, {seq_len=}")

        # mask support
        if use_mask and mask_in is not None:
            assert isinstance(mask_in, (torch.Tensor,)) and 1 <= mask_in.dim() <=4
            extra_dims = 4  - mask_in.dim()
            mask = mask_in.reshape((1,) * extra_dims + mask_in.shape)
            q0, q1, q2, q3 = q.shape
            mask = mask.expand(q0, q1, q2, k.shape[2])
            mask_strides = mask.stride()
        else:
            mask_strides = (None,)*4

        _fwd_kernel[grid](q, k, v, 
                          output,
                          qk_scale_factor,
                          softmax_normalizer,
                          use_causal, #=use_causal,
                          use_mask, #=use_mask,
                          mask, # reshaped from mask_in, alibi mask
                          # 4d strides,
                          q.stride(0), # batch
                          q.stride(1), # num heads
                          q.stride(2), # seq len
                          q.stride(3), # head dim
                          k.stride(0), # batch
                          k.stride(1), # num heads
                          k.stride(2), # seq len
                          k.stride(3), # head dim
                          v.stride(0), # batch
                          v.stride(1), # num heads
                          v.stride(2), # seq len
                          v.stride(3), # head dim
                          *mask_strides,
                          output.stride(0), # batch
                          output.stride(1), # num heads
                          output.stride(2), # seq len
                          output.stride(3), # head dim

                          block_m = block_m,
                          block_n = block_n,
                          block_head_dim = kdim,
                          num_heads = num_heads, 
                          seq_len = seq_len,
                          
                          # special params - absorbed by triton
                          num_warps=num_warps,
                          num_stages=num_stages,
                          )
        
        # save for backwards
        ctx.save_for_backward(q, k, v, mask, output, softmax_normalizer, )
        ctx.grid = grid
        ctx.qk_scaling = qk_scale_factor
        ctx.head_dim = kdim
        ctx.use_causal = use_causal
        ctx.use_mask = use_mask

        return output
    
    @staticmethod
    def backward(ctx, do):
        block_size = 128 
        print(f"in backward -- saved tensors:")
        unpack = ctx.saved_tensors
        # for item in unpack:
        #     print(f"{item.shape=}")
        '''item.shape=torch.Size([2, 64, 4096, 64])
        item.shape=torch.Size([2, 64, 4096, 64])
        item.shape=torch.Size([2, 64, 4096, 64])
        item.shape=torch.Size([2, 64, 4096, 4096])
        item.shape=torch.Size([2, 64, 4096, 64])
        item.shape=torch.Size([128, 4096])
        '''
        q, k, v, mask, output, softmax_normalizer = unpack
        
        grid = ctx.grid
        grid_n = grid[1]

        do = do.contiguous()
        
        # seq parallel TODO
        seq_parallel=False
        if seq_parallel:
            raise ValueError("seq parallel not implemented yet")
        else:
            dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(softmax_normalizer)

        if mask is not None:
            mask_strides = mask.stride()
        else:
            mask_strides = (None,)*4
        
        seq_len = q.shape[2]
        preprocess_grid = (cdiv(seq_len, block_size) * grid_n,)

        _bwd_preprocess[preprocess_grid](
            output,
            do,
            delta,
            block_m=block_size,
            dim_head=ctx.head_dim,
        )

        seq_len_kv = k.shape[2]
        bwd_grid = (grid_n, 1) if not seq_parallel else (grid_n, cdiv(seq_len_kv, block_size))
        
        _bwd_kernel[bwd_grid](
            q, k, v,
            ctx.qk_scaling,
            mask,
            output,
            do,
            dq,
            dk,
            dv,
            softmax_normalizer,
            delta,
            output.numel(),
            # strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            *mask_strides,


        )
        # dummy vals
        dq = dk = dv = torch.ones_like(do)
        return dq, dk, dv, None, None, None, None

new_flash2 = _newattention.apply





                                                            
