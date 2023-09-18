import pytest
import torch
import triton
import time

from torch.nn.functional import  scaled_dot_product_attention as flash_sdpa
from new_flash2_alibi import new_flash2 as attention
from base_flash2 import attention as prev_triton_attn
@pytest.mark.parametrize("batch, num_heads, seq_len, dim_head", [(4, 48, 512, 64),
                                                                 (2, 48, 4096, 64 ),#
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

    dout = torch.ones_like(q)

    use_causal = False
    use_mask = False
    run_seq_parallel = True

    run_base = True
    run_prev_flash = False
    
    run_sdpa = True

    if use_mask:
        mask = torch.ones((seq_len, seq_len), device=q.device, dtype=q.dtype)
    else:
        mask = None

    qk_scale = k.shape[-1]**0.5
    #sm_scale.to(torch.bfloat16)
    start = time.perf_counter()
    # params = q k v scaling use_causal
    # for i in range(0,1000):

    tri_out_nosq = attention(q,k,v,None, use_causal, use_mask, 
                        mask, False) # qk_scale)
    stop = time.perf_counter()
    triton_time_nosq = round(stop-start, 4)
    print(f"triton fwd no_sq compute time = {triton_time_nosq}")
    
    
    start = time.perf_counter()
    tri_out_yessq = attention(q,k,v,None, use_causal, use_mask, 
                        mask, True) # qk_scale)
    print(f"triton fwd yes_sq compute time = {tri_out_yessq}")
    stop = time.perf_counter()
    triton_time = round(stop-start, 4)

    print(f"{tri_out_nosq[0][0][0]=}")
    torch.testing.assert_close(tri_out_nosq, tri_out_yessq, atol=0, rtol=1e-4)
    # params: q,k,v,causal,sm_scale,
        # mask: torch.Tensor = None,
        # sequence_parallel=False,
    
    if run_prev_flash:
        prev_fwd_out = prev_triton_attn(q,k,v,use_causal, 
                             qk_scale, mask, run_seq_parallel)
    
        print(f"{prev_fwd_out[0][0][0]=}")
    
    # --- sdpa -----
    if run_sdpa:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        sdpa_fwd_out = flash_sdpa(q,k,v,attn_mask=mask, is_causal = use_causal, scale=qk_scale)
        print(f"{sdpa_fwd_out[0][0][0]=}")


        # manual
        ##qk_scaling = qk_scale # * 1.44269504
        #q = q*qk_scaling # .to(k.dtype.element_ty)
        # mha = torch.matmul(q, k.transpose(2,3)) * qk_scale
        # mha = torch.softmax(mha.float(), dim=-1).to(dtype)
        # TODO - need to add causal mask for regular calc
        # expected_out = torch.matmul(mha, v)
        #if not use_causal:
        # print(f"{expected_out[0][0][0]=}")
        if run_prev_flash:
            torch.testing.assert_close(prev_fwd_out, tri_out, rtol=0, atol=1e-1)
        #torch.testing.assert_close(tri_out, expected_out, rtol=0, atol=1e-1)




    def clear_grads(*args):
        for item in args:
            item.grad=None

    # ====== backward ===============
    tri_out_nosq.backward(dout)
    #if run_base:
    #    base_out.backward(dout)
    #sdpa_out.backwar

    # derivatives
    tri_dv = v.grad.clone().detach()
    tri_dk = k.grad.clone().detach()
    tri_dq = q.grad.clone().detach()

    clear_grads(q,k,v)

    tri_out_yessq.backward(dout)
    #if run_base:
    #    base_out.backward(dout)
    #sdpa_out.backwar

    # derivatives
    trisq_dv = v.grad.clone().detach()
    trisq_dk = k.grad.clone().detach()
    trisq_dq = q.grad.clone().detach()

    clear_grads(q,k,v)


    

    if run_prev_flash:
        prev_fwd_out.backward(dout)
        prevt_dv = v.grad.clone()
        prevt_dk = k.grad.clone()
        prevt_dq = q.grad.clone()

    clear_grads(q,k,v)
    
    if run_sdpa:
        sdpa_fwd_out.backward(dout)
        sdpa_dv = v.grad.clone()
        sdpa_dk = k.grad.clone()
        sdpa_dq = q.grad.clone()



    print(f"{tri_dv[0][0][10]=}")
    print(f"{trisq_dv[0][0][10]}=")
    
    if run_prev_flash:
        print(f"{prevt_dv[0][0][10]=}")
    if run_sdpa:
        print(f"{sdpa_dv[0][0][10]=}")
        print(f"type sdpa {sdpa_dv.dtype=}")
    
    print(f"==== dk ============")
    print(f"{tri_dk[0][0][0]=}")
    print(f"{trisq_dk[0][0][0]}=")
    if run_prev_flash:
        print(f"{prevt_dk[0][0][0]=}")
    if run_sdpa:
        print(f"{sdpa_dk[0][0][0]=}")
    
    torch.testing.assert_close(tri_dk, tri_dk, rtol=1e-1, atol=1e-1)

