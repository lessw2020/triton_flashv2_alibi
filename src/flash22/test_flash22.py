# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# test the trition flash22 impl with alibi

import pytest
import torch

import triton
import triton.language as tl
import time

from torch.nn.functional import  scaled_dot_product_attention as flash_sdpa
from triton_flash22alibi import flash22attention
from orig_flash22 import attention as orig22_attention

'''
def track_timing(core_function, msg):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = core_function(*args, **kwargs)
        stop = time.perf_counter()
        time_taken = round(stop-start, 4)
        print(f"{msg}: {time_taken=}")
        return res
'''
@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(2, 64, 4096, 64 ),#
                                                                 (2, 48, 512, 16),
        (4, 32, 4096, 64),
        (4, 48, 2048, 64),
       (4, 48, 1024, 128),])
@pytest.mark.parametrize("dtype",[torch.float16])


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

    dout = torch.ones_like(q)

    use_causal = True

    qk_scale = k.shape[-1]**0.5

    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device='cuda'))


    # ---- flash 22 -----
    tri_fwd_time = None
    start = time.perf_counter()
    tri_out = flash22attention(q, k, v, use_causal, qk_scale)
    stop = time.perf_counter()
    tri_fwd_time = round(stop-start, 4)
    print(f"{tri_out[0][0][0][0:2]=}")

    # 2nd run - flash 22
    # ---- flash 22 -----
    tri_fwd_time2 = None
    start = time.perf_counter()
    tri_out2 = flash22attention(q, k, v, use_causal, qk_scale)
    stop = time.perf_counter()
    tri_fwd_time2 = round(stop-start, 4)
    print(f"{tri_out2[0][0][0][0:2]=}")

    # original flash22
    orig_fwd_time = None
    start = time.perf_counter()
    orig_out = orig22_attention(q,k,v,use_causal, qk_scale)
    stop = time.perf_counter()
    orig_fwd_time = round(stop-start, 4)
    print(f"{orig_out[0][0][0][0:2]=}")

    # --- sdpa -----
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    sdpa_fwd_time = None
    start = time.perf_counter()
    sdpa_out = flash_sdpa(q,k,v,attn_mask=None, is_causal = use_causal, scale=qk_scale)
    stop = time.perf_counter()
    sdpa_fwd_time = round(stop-start, 4)
    print(f"{sdpa_out[0][0][0]=}")

    # manual
    mha_fwd_time = None
    start = time.perf_counter()

    mha_scores = torch.matmul(q, k.transpose(2,3)) * qk_scale
    if use_causal:
        mha_scores[:, :, causal_mask==0] = float("-inf")
    probs = torch.softmax(mha_scores.float(), dim=-1).half()
    ref_out = torch.matmul(probs,v)
    stop = time.perf_counter()
    mha_fwd_time = round(stop-start, 4)
    print(f"{ref_out[0][0][0]=}")


    # timing
    print(f"Fwd Timing: {mha_fwd_time=}, {tri_fwd_time=}, {tri_fwd_time2=} {orig_fwd_time=}, {sdpa_fwd_time=}")

    # testing
    torch.testing.assert_close(ref_out, sdpa_out, rtol=0, atol=1e-1)

