import torch
from triton import cdiv, jit  
import triton.language as tl

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, mask=None, sequence_parallel=False):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80"
            )
        BLOCK_M = 128
        BLOCK_N = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        num_warps = 4 if Lk <= 64 else 8

        has_mask = mask is not None
        if has_mask:
            assert isinstance(mask, torch.Tensor) and 1 <= mask.dim() <= 4
            extra_dims = 4 - mask.dim()
            mask = mask.reshape((1,) * extra_dims + mask.shape)
            mask = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            mask_strides = mask.stride()
        else:
            mask_strides = (None,) * 4

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            mask,
            L,
            o,
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
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            HAS_MASK=has_mask,
            num_warps=num_warps,
            num_stages=4,
        )

        ctx.save_for_backward(q, k, v, mask, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        return o