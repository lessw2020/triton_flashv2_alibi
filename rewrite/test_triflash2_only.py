import pytest
import torch
import triton
import time

from torch.nn.functional import  scaled_dot_product_attention as flash_sdpa
from new_flash2_alibi import new_flash2 as attention
from base_flash2 import attention as orig_attn
@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(2, 64, 4096, 64 ),#
                                                                 #(2, 48, 512, 16),
        # (4, 48, 1024, 32),
        # (4, 48, 1024, 64),
    ],)  #  (4, 48, 1024, 128)]])
@pytest.mark.parametrize("dtype",[torch.bfloat16])


def test_fwd_attention(batch, num_heads, seq_len, dim_head, dtype):
    torch.manual_seed(2020)

    def make_qkv(batch, num_heads, seq_len, dim_head, dtype):
        """produces consistent qkv's"""
        res = torch.empty((batch, num_heads, seq_len, dim_head), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
        res.requires_grad_()
        return res


    q = make_qkv(batch, num_heads, seq_len, dim_head, dtype)
    k = make_qkv(batch, num_heads, seq_len, dim_head, dtype)
    v = make_qkv(batch, num_heads, seq_len, dim_head, dtype)

    assert id(q) != id(k)  

    dout = torch.randn_like(q)

    use_causal = True
    use_mask = True

    if use_mask:
        mask = torch.ones((seq_len, seq_len), device=q.device, dtype=q.dtype)
    else:
        mask = None

    qk_scale = k.shape[-1]**0.5
    #sm_scale.to(torch.bfloat16)
    start = time.perf_counter()
    # params = q k v scaling use_causal
    # for i in range(0,1000):

    tri_out = attention(q,k,v,None, use_causal, use_mask, mask) # qk_scale)
    stop = time.perf_counter()
    triton_time = round(stop-start, 4)
    print(f"triton compute time = {triton_time}")
    # params: q,k,v,causal,sm_scale,
        # mask: torch.Tensor = None,
        # sequence_parallel=False,
    base_out = orig_attn(q,k,v,use_causal, qk_scale, mask)
    
    print(f"{tri_out[0][0][0]=}")
    print(f"{base_out[0][0][0]=}")
    
    # --- sdpa -----
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #sdpa_out = flash_sdpa(q,k,v,attn_mask=mask, is_causal = use_causal, scale=qk_scale)
    #print(f"{sdpa_out[0][0][0]=}")


    # manual
    ##qk_scaling = qk_scale # * 1.44269504
    #q = q*qk_scaling # .to(k.dtype.element_ty)
    mha = torch.matmul(q, k.transpose(2,3)) * qk_scale
    mha = torch.softmax(mha.float(), dim=-1).to(dtype)
    # TODO - need to add causal mask for regular calc
    expected_out = torch.matmul(mha, v)
    #if not use_causal:
    print(f"{expected_out[0][0][0]=}")

    torch.testing.assert_close(base_out, tri_out, rtol=0, atol=1e-2)
    #torch.testing.assert_close(tri_out, expected_out, rtol=0, atol=1e-1)




    def clear_grads(*args):
        for item in args:
            item.grad=None

    # ====== backward ===============
    #tri_out.backward(dout)
    base_out.backward(dout)
    #sdpa_out.backwar

    # derivatives
    '''tri_dv = v.grad.clone().detach()
    tri_dk = k.grad.clone().detach()
    tri_dq = q.grad.clone().detach()

    clear_grads(q,k,v)
    '''

    base_out.backward(dout)
    base_dv = v.grad.clone()
    base_dk = k.grad.clone()
    base_dq = q.grad.clone()

    clear_grads(q,k,v)
    




    #print(f"{tri_dv[0][0][0]=}")
    print(f"{base_dv[0][0][0]=}")
    #print(f"{tri_dk[0][0][0]=}")
    print(f"{base_dk[0][0][0]=}")

