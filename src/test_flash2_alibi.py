import pytest
import torch

import triton

import triton.ops

from alibi_flashv2 import flash2_alibi
from base_flash2 import attention as baseAttention
from pr_alibi import attention as pr_Attention


def _banded_sq_matrix(size, value=1, device=None, dtype=None):
    """Return a square matrix with some bands filled with the given value."""
    mat = torch.full((size, size), value, device=device, dtype=dtype)
    for i in range(-size + 1, size, 3):
        torch.diagonal(mat, offset=i)[:] = 0
    return mat


masked = False


@pytest.mark.parametrize(
    "Z, H, N_CTX, D_HEAD",
    [
        (2, 48, 512, 16),
        # (4, 48, 1024, 32),
        # (4, 48, 1024, 64),
    ],  #  (4, 48, 1024, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("masked", [False])
@pytest.mark.parametrize("seq_par", [False, True])
def test_op(Z, H, N_CTX, D_HEAD, dtype, causal, masked, seq_par):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    if causal:
        M = torch.triu(
            torch.full(
                # Cannot give dtype here due to BF16 incompatibility with torch<=2.0:
                # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
                (N_CTX, N_CTX),
                float("-inf"),
                device="cuda",
            ),
            diagonal=1,
        ).to(dtype)
    else:
        M = torch.zeros((N_CTX, N_CTX), device="cuda", dtype=dtype)

    if masked:
        # Use an interesting mask but let the Triton kernel handle causality.
        M_triton = _banded_sq_matrix(N_CTX, float("-inf"), dtype=dtype, device="cuda")
        # Keep causality setting for the reference calculation.
        M = torch.where(
            M == float("-inf"),
            M,
            M + M_triton,
        )
    else:
        M_triton = None

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = p + M
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    # tri_out = flash2_alibi(q, k, v, causal, sm_scale, M_triton, seq_par)
    # tri_out = triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    tri_out = baseAttention(q, k, v, causal, sm_scale, M_triton, seq_par)
    # tri_out = pr_Attention(q, k, v, causal, sm_scale, M_triton, seq_par)
    print(f"{ref_out.shape=}, {ref_out[0][0][0][0]=}")
    print(f" {tri_out.shape=}, {tri_out[0][0][0][0]=}")
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)
