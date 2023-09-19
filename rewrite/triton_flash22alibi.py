# learning the new flash2 impl from Triton team

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_inner(
    accum, l_i, m_i, 
    q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_headdim: tl.constexpr,
    stage: tl.constexpr,
    offsets_m: tl.constexpr,
    offsets_n: tl.constexpr,

):
    if stage ==1:
        low, high = 0, start_m * block_m
    else:
        low = start_m * block_m
        high = (start_m +1) * block_m
        # tell compiler about low to block_m relationship
        low = tl.multiple_of(low, block_m)

    K_block_ptr = tl.advance(K_block_ptr, (0,low))
    V_block_ptr = tl.advance(V_block_ptr, (low,0))

    # loop over k, v and update accum
    for start_n in range(low, high, block_n):
        start_n = tl.multiple_of(start_n, block_n)
        # -- qk compute --
        k = tl.load(K_block_ptr)
        qk = tl.zeros([block_m, block_n], dtype = tl.float32)
        qk += tl.dot(q,k)

        if stage ==2:
            mask = offsets_m[:,None] >= (start_n + offsets_n[None,:])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk,1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk,1) * qk_scale)
            qk = qk * qk_scale - m_ij[:,None]

        probs = tl.math.exp2(qk)
        l_ij = tl.sum(probs,1)
        # -- update l
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # update output accum
        accum = accum * alpha[:, None]
        v = tl.load(V_block_ptr)
        accum += tl.dot(probs.to(tl.float16), v)
        # update m
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (block_n,0))
        K_block_ptr = tl.advance(K_block_ptr, (0, block_n))
    return accum, l_i, m_i
        


