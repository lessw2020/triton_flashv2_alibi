import pytest
import torch
import triton

from new_flash2_alibi import new_flash2 as attention
from base_flash2 import attention as orig_attn
@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(1, 2, 64, 32, ),])
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

    sm_scale = 0.5

    tri_out = attention(q,k,v,sm_scale)
    base_out = orig_attn(q,k,v,False, sm_scale)

    print(f"{tri_out[0][0][0][0:5]=}")
    print(f"{base_out[0][0][0][0:5]=}")

    # manual
    mha = torch.matmul(q, k.transpose(2,3)) * sm_scale
    mha = torch.softmax(mha.float(), dim=-1).to(dtype)

    expected_out = torch.matmul(mha, v)
    print(f"{expected_out[0][0][0][0:5]=}")





    # ====== backward ===============
    tri_out.backward(dout)

    # derivatives
    tri_dv = v.grad.clone()
    tri_dk = k.grad.clone()
    tri_dq = q.grad.clone()
    # print(f"{tri_dv=}")