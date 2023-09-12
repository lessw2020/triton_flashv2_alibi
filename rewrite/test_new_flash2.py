import pytest
import torch
import triton

from new_flash2_alibi import new_flash2 as attention

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