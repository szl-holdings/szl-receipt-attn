# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Original SZL Triton tiles + online softmax. From the FA papers, not from Dao CUDA."""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:  # pragma: no cover
    raise ImportError("triton is required for the CUDA path") from e


@triton.jit
def _szl_tiled_attn_fwd(
    Q, K, V, O,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_oh, stride_ot, stride_od,
    T_Q, T_KV, D,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + bh * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < T_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for kv_start in range(0, T_KV, BLOCK_N):
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        k_ptrs = K + bh * stride_kh + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
        v_ptrs = V + bh * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < T_KV) & (offs_d[:, None] < D), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < T_KV) & (offs_d[None, :] < D), other=0.0)
        qk = tl.dot(q, k) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        qk = tl.where(offs_n[None, :] < T_KV, qk, -float("inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = O + bh * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), mask=q_mask)


def triton_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 4:
        raise ValueError("expected [B, H, T, D]")
    b, h, tq, d = q.shape
    tkv = k.shape[2]
    if d > 32:
        raise NotImplementedError("v0 Triton path supports head dim <= 32")
    sm = (d ** -0.5) if scale is None else scale
    q2 = q.contiguous().reshape(b * h, tq, d)
    k2 = k.contiguous().reshape(b * h, tkv, d)
    v2 = v.contiguous().reshape(b * h, tkv, d)
    o2 = torch.empty_like(q2)
    BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, 32
    grid = (triton.cdiv(tq, BLOCK_M), b * h)
    _szl_tiled_attn_fwd[grid](
        q2, k2, v2, o2,
        q2.stride(0), q2.stride(1), q2.stride(2),
        k2.stride(0), k2.stride(1), k2.stride(2),
        v2.stride(0), v2.stride(1), v2.stride(2),
        o2.stride(0), o2.stride(1), o2.stride(2),
        tq, tkv, d, sm,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o2.reshape(b, h, tq, d)
