# rewrite flash2 with alibi support, from scratch and first principles.
# 

# step 1 - get forward kernel going with online softmax.

import torch

from triton import cdiv, jit
from triton import language as tl

_supported_head_dims = (16,32,64,128)

'''
    
    
    block_m: tl.constexpr,
    block_dim_model: tl.constexpr,
    # num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    '''
@jit
def _fwd_kernel(
    Q: torch.tensor, 
    K: torch.tensor, 
    V: torch.tensor,
    k_sqrt_scale_factor: torch.float,
    softmax_normalizer: torch.tensor, 
    *,
    Out: torch.tensor,
    num_heads: int,
    seq_len: int,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_head_dim: tl.constexpr,
    use_causal: tl.constexpr, #: bool,
    use_mask: tl.constexpr, #: bool, 

    

    ):
    start_m = tl.program_id(0) # row offset(?)
    off_hz = tl.program_id(1) # batch offset


    
class _newattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_causal=True, use_mask = False):
        qdim, kdim, vdim = q.shape[-1], k.shape[-1], v.shape[-1]
        print(f"{qdim=}")
        # confirm suitable qkv shapes
        assert qdim == kdim and kdim == vdim
        assert kdim in _supported_head_dims

        # currently support only mask or only causal (mask should include causal)
        assert use_causal != use_mask, f"use causal {use_causal=} and {use_mask=} are mutually exclusive"

        # block tuning
        block_m = 128
        block_n = 64
        print(f"block sizes: {block_m=}, {block_n=}")

        output = torch.empty_like(q)

        k_sqrt_scale_factor = kdim**0.5  
        # triton tuning
        num_warps = 4 if kdim <= 64 else 8
        num_stages = 4

        softmax_normalizer = torch.empty((q.shape[0]*q.shape[1], q.shape[2]),device=q.device, dtype=torch.float32)
        
        print(f"{softmax_normalizer.shape=}")
        grid = (cdiv(q.shape[2], block_m ), q.shape[0] * q.shape[1],1)
        print(f"{grid=}")
        num_heads, seq_len = q.shape[1], q.shape[2]
        print(f"{num_heads=}, {seq_len=}")

        _fwd_kernel[grid](q, k, v, 
                          k_sqrt_scale_factor,
                          Out=output,
                          softmax_normalizer = softmax_normalizer,
                          block_m = block_m,
                          block_n = block_n,
                          block_head_dim = kdim,
                          num_heads = num_heads, 
                          seq_len = seq_len,
                          use_causal=use_causal,
                          use_mask=use_mask,
                          # special params - absorbed by triton
                          num_warps=num_warps,
                          num_stages=num_stages,
                          )
        

        ctx.save_for_backward(q, k, v, output, softmax_normalizer, )
        return output
    
    @staticmethod
    def backward(ctx, do):
        block = 128  
        print(f"in backward")
        # dummy vals
        dq = dk = dv = torch.ones_like(do)
        return dq, dk, dv, None

new_flash2 = _newattention.apply





                                                            
