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
    block_m: tl.constexpr,
    block_dim_model: tl.constexpr,
    

    ):
    start_m = tl.program_id(0) # row offset(?)
    off_hz = tl.program_id(1) # batch offset


    
class _newattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        qlen, klen, vlen = q.shape[-1], k.shape[-1], v.shape[-1]
        print(f"{qlen=}")
        block_m = 128
        block_n = 64

        assert qlen == klen and klen == vlen
        assert klen in _supported_head_dims

        block_dim_model = klen  # model dimensionality

        output = torch.empty_like(q)
        k_sqrt_scale_factor = klen**0.5  
        # triton tuning
        num_warps = 4 if klen <= 64 else 8
        num_stages = 4

        softmax_normalizer_meta = torch.empty((q.shape[0]*q.shape[1], q.shape[2]),device=q.device, dtype=torch.float32)
        
        print(f"{softmax_normalizer_meta.shape=}")
        grid = (cdiv(q.shape[2], block_m ), q.shape[0] * q.shape[1],1)
        print(f"{grid=}")

        _fwd_kernel[grid](q, k, v, 
                          k_sqrt_scale_factor,
                          Out=output,
                          softmax_normalizer = softmax_normalizer_meta,
                          block_m = block_m,
                          block_dim_model = block_dim_model,
                          # special params - absorbed by triton
                          num_warps=num_warps,
                          num_stages=num_stages,
                          )
        

        ctx.save_for_backward(q, k, v)
        return output
    
    @staticmethod
    def backward(ctx, do):
        block = 128  
        print(f"in backward")
        # dummy vals
        dq = dk = dv = torch.ones_like(do)
        return dq, dk, dv, None

new_flash2 = _newattention.apply





                                                            
