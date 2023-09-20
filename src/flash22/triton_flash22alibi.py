# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This version adds Alibi / Attention mask support
# builds on:

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_inner(
    accum, l_i, m_i, 
    q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_headdim: tl.constexpr,
    stage: tl.constexpr,
    offsets_m: tl.constexpr,
    offsets_n: tl.constexpr,

):
    if stage ==1:
        low, high = 0, start_m * block_m
    else:
        low = start_m * block_m
        high = (start_m +1) * block_m
        # tell compiler about low to block_m relationship
        low = tl.multiple_of(low, block_m)

    K_block_ptr = tl.advance(K_block_ptr, (0,low))
    V_block_ptr = tl.advance(V_block_ptr, (low,0))

    # loop over k, v and update accum
    for start_n in range(low, high, block_n):
        start_n = tl.multiple_of(start_n, block_n)
        # -- qk compute --
        k = tl.load(K_block_ptr)
        qk = tl.zeros([block_m, block_n], dtype = tl.float32)
        qk += tl.dot(q,k)

        if stage ==2:
            mask = offsets_m[:,None] >= (start_n + offsets_n[None,:])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk,1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk,1) * qk_scale)
            qk = qk * qk_scale - m_ij[:,None]

        probs = tl.math.exp2(qk)
        l_ij = tl.sum(probs,1)
        # -- update l
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # update output accum
        accum = accum * alpha[:, None]
        v = tl.load(V_block_ptr)
        accum += tl.dot(probs.to(tl.float16), v)
        # update m
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (block_n,0))
        K_block_ptr = tl.advance(K_block_ptr, (0, block_n))
    return accum, l_i, m_i

@triton.jit
def _attn_fwd(
    Q,K,V, sm_scale, M, Out, mask_in,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mz, stride_mh, stride_mm, stride_mn,
    Z, H, 
    seq_len: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_headdim: tl.constexpr,
    stage: tl.constexpr,

):
    start_m = tl.program_id(0)
    offset_hz = tl.program_id(1)
    offset_z = offset_hz // H
    offset_h = offset_hz % H

    offset_z.to(tl.int64)
    offset_h.to(tl.int64)

    qkv_offset = offset_z * stride_qz + offset_h * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape = (seq_len, block_headdim),
        strides = (stride_qm, stride_qk),
        offsets = (block_m, block_headdim),
        block_shape=(block_m, block_headdim),
        order = (1,0),

    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(seq_len, block_headdim),
        strides=(stride_vk, stride_vn), 
        offsets=(0, 0),
        block_shape=(block_n, block_headdim),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(block_headdim, seq_len),
        strides=(stride_kk, stride_kn), # reversed
        offsets=(0, 0),
        block_shape=(block_headdim, block_n),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(seq_len, block_headdim),
        strides=(stride_om, stride_on),
        offsets=(start_m * block_m, 0),
        block_shape=(block_m, block_headdim),
        order=(1, 0),
    )

    offsets_m = start_m * block_m + tl.arange(0,block_m)
    offsets_n = tl.arange(0, block_n)

    # init softmax metas and accumulator
    m_i = tl.zeros([block_m], dtype =tl.float32) - float("inf")
    l_i = tl.zeros([block_m], dtype=tl.float32)+1.0
    acc = tl.zeros([block_m, block_headdim], dtype = tl.float32)

    #qk scaling
    # qk_scale = sm_scale   # ??
    qk_scale = sm_scale * 1.44269504 # 1/log(2)

    q = tl.load(Q_block_ptr)
    # stage 1 == off_band
    if stage & 1:
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q, 
            K_block_ptr, V_block_ptr,
            start_m, qk_scale, 
            block_m, block_n, block_headdim,
            1, offsets_m, offsets_n
        )
    # barrier hints compiler it can schedule dual loops independently
    tl.debug_barrier()
    # stage 2: on_band
    if stage & 2:
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            block_m, block_n, block_headdim,
            2, offsets_m, offsets_n,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:,None]
    m_ptrs = M + offset_hz * seq_len + offsets_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

empty = torch.empty(128, device="cuda")

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, attn_mask=None):
        #verify shape
        qlen, klen, vlen = q.shape[-1], k.shape[-1], v.shape[-1]
        assert qlen == klen and klen == vlen
        assert klen in {16, 32, 64, 128}
        q_batch, q_numheads, q_seqlen, q_headdim = q.shape

        out = torch.empty_like(q)
        block_m = 128
        block_n = 64 if klen <=64  else 32
        num_stages = 4 if klen <=64 else 3
        num_warps = 4
        row_chunks = triton.cdiv(q_seqlen, block_m)
        grid = (row_chunks, q_batch * q_numheads,1)

        M = torch.empty((q_batch, q_numheads, q_seqlen), device=q.device, dtype = torch.float32)
        use_mask = attn_mask is not None
        if use_mask:
            assert isinstance(attn_mask, (torch.Tensor,)) and 1 <= attn_mask.dim() <=4
            extra_dims = 4 - attn_mask.dim()
            mask_shape = (1,) * extra_dims + attn_mask.shape[:-1] + (k.shape[2],)
            mask = attn_mask.view(*mask_shape).expand_as(k)
            mask_strides = mask.stride()
        else:
            mask=None
            mask_strides = (None,)*4

        _attn_fwd[grid](
            q, k, v, sm_scale, M, out, mask,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mask_strides[0], mask_strides[1], 
            mask_strides[2], mask_strides[3],
            q_batch, q_numheads,
            q_seqlen,
            block_m,
            block_n, 
            klen,
            stage=3,
            num_warps=num_warps,
            num_stages=num_stages,


        )

        ctx.save_for_backward(q, k, v, out, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.block_headdim = klen
        ctx.causal = causal
        return out
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, out, M = ctx.saved_tensors
        print(f"in backward - not implemented yet")

        return None, None, None, None, None

flash22attention = _attention.apply




