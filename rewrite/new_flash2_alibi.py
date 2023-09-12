# rewrite flash2 with alibi support, from scratch and first principles.
# 

# step 1 - get forward kernel going with online softmax.

import torch

from triton import cdiv, jit
from triton import language as tl

@jit
def _fwd_kernel(
    Q, K, V, Out
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
        output = torch.empty_like(q)

        grid = (cdiv(q.shape[2], ))
        _fwd_kernel[grid](q, k, v)

        ctx.save_for_backward(q, k, v)
        return output
    
    @staticmethod
    def backward(ctx, do):
        block = 128  
        print(f"in backward")
        return None

new_flash2 = _newattention.apply





                                                            
