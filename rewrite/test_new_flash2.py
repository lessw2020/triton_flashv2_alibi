import pytest
import torch
import triton

from new_flash2_alibi import new_flash2 as attention

@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(1, 2, 64, 32, ),])
@pytest.mark.parametrize("dtype",[torch.bfloat16])

def test_attention(batch, num_heads, seq_len, dim_head, dtype):
    torch.manual_seed(2020)

    q = torch.empty((batch, num_heads, seq_len, dim_head), dtype=dtype, device="cuda")
    q.normal_(mean=0.0, std=0.5)
    q.requires_grad_()

    print(f"{q=}")