import pytest
import torch
import triton
import time

from torch.nn.functional import  scaled_dot_product_attention as flash_sdpa
from new_flash2_alibi import new_flash2 as attention
from base_flash2 import attention as orig_attn
@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(4, 64, 1024, 64 ),#
                                                                 #(2, 48, 512, 16),
        # (4, 48, 1024, 32),
        # (4, 48, 1024, 64),
    ],)  #  (4, 48, 1024, 128)]])
@pytest.mark.parametrize("dtype",[torch.bfloat16])


def test_attention(batch, num_heads, seq_len, dim_head, dtype):
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


    print(f"{q=}")
    dout = torch.randn_like(q)

    qk_scale = k.shape[-1]**0.5
    #sm_scale.to(torch.bfloat16)
    start = time.perf_counter()
    tri_out = attention(q,k,v,) # qk_scale)
    stop = time.perf_counter()
    triton_time = round(stop-start, 4)
    print(f"triton compute time = {triton_time}")
    base_out = orig_attn(q,k,v,False, qk_scale)
    
    print(f"{tri_out[0][0][0]=}")
    print(f"{base_out[0][0][0]=}")
    
    # --- sdpa -----
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    sdpa_out = flash_sdpa(q,k,v,scale=qk_scale)
    print(f"{sdpa_out[0][0][0]=}")


    # manual
    ##qk_scaling = qk_scale # * 1.44269504
    #q = q*qk_scaling # .to(k.dtype.element_ty)
    mha = torch.matmul(q, k.transpose(2,3)) * qk_scale
    mha = torch.softmax(mha, dim=-1).to(dtype)

    expected_out = torch.matmul(mha, v)
    print(f"{expected_out[0][0][0]=}")

    print(f"{tri_out[0, 25, 246, 62]=}")
    print(f"{base_out[0, 25, 246, 62]=}")
    print(f"{sdpa_out[0, 25, 246, 62]=}")
    print(f"{expected_out[0, 25, 246, 62]=}")
    #torch.testing.assert_close(base_out, sdpa_out, rtol=0, atol=1e-2)
    torch.testing.assert_close(tri_out, base_out, rtol=0, atol=1e-2)





    # ====== backward ===============
    tri_out.backward(dout)

    # derivatives
    tri_dv = v.grad.clone()
    tri_dk = k.grad.clone()
    tri_dq = q.grad.clone()
    # print(f"{tri_dv=}")