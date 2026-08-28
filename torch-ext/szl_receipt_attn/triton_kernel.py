# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Original Triton tiled fused-attention kernel (CUDA JIT path).

Written from the FlashAttention papers' tile + online-softmax silhouette:

* FA1 https://arxiv.org/abs/2205.14135
* FA2 https://arxiv.org/abs/2307.08691
* FA3 https://arxiv.org/abs/2407.08608

This is an original SZL construction in the fused-attention category.
It is NOT a rehost of Dao-AILab/flash-attention or kernels-community
flash-attn2/3/4 (no CUDA, no CUTLASS, no CuTe).

Triton runs on CUDA. This module is imported only when that path is
selected. No speedup is claimed.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .const import TRITON_MAX_HEAD_DIM, TRITON_MIN_HEAD_DIM
from .reference import broadcast_attn_mask


def _next_pow2(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


@triton.jit
def _szl_tiled_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    mask_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    seq_q,
    seq_k,
    n_q_heads,
    n_kv_heads,
    head_dim,
    sm_scale,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """One program: one query tile of one (batch, head). Inner loop: KV tiles."""
    q_tile = tl.program_id(0)
    batch_head = tl.program_id(1)

    batch = batch_head // n_q_heads
    q_head = batch_head % n_q_heads
    kv_group = n_q_heads // n_kv_heads
    kv_head = q_head // kv_group

    q_offs = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_DIM)
    q_valid = q_offs < seq_q
    d_valid = d_offs < head_dim

    q_ptrs = (
        q_ptr
        + batch * stride_qb
        + q_head * stride_qh
        + q_offs[:, None] * stride_qm
        + d_offs[None, :] * stride_qd
    )
    # FA2 silhouette: fold the scale into Q so the KV loop is matmul + softmax.
    query = tl.load(q_ptrs, mask=q_valid[:, None] & d_valid[None, :], other=0.0)
    query = (query.to(tl.float32) * sm_scale).to(query.dtype)

    running_m = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    running_l = tl.zeros([BLOCK_Q], dtype=tl.float32)
    out_acc = tl.zeros([BLOCK_Q, BLOCK_DIM], dtype=tl.float32)

    kv_tiles = tl.cdiv(seq_k, BLOCK_KV)
    for kv_tile in range(0, kv_tiles):
        k_offs = kv_tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
        k_valid = k_offs < seq_k

        k_ptrs = (
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + k_offs[:, None] * stride_kn
            + d_offs[None, :] * stride_kd
        )
        key = tl.load(k_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0)

        scores = tl.dot(query, tl.trans(key))
        scores = scores.to(tl.float32)

        keep = q_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            keep = keep & (k_offs[None, :] <= q_offs[:, None])

        if HAS_MASK:
            mask_ptrs = (
                mask_ptr
                + batch * stride_mb
                + q_head * stride_mh
                + q_offs[:, None] * stride_mm
                + k_offs[None, :] * stride_mn
            )
            mask_tile = tl.load(
                mask_ptrs, mask=q_valid[:, None] & k_valid[None, :], other=0
            )
            if MASK_IS_BOOL:
                keep = keep & (mask_tile != 0)
            else:
                scores = scores + mask_tile.to(tl.float32)

        scores = tl.where(keep, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_m = tl.maximum(running_m, tile_max)
        finite = new_m > -float("inf")
        alpha = tl.exp(running_m - new_m)
        alpha = tl.where(finite, alpha, 0.0)

        weights = tl.exp(scores - new_m[:, None])
        weights = tl.where(keep & finite[:, None], weights, 0.0)

        running_l = running_l * alpha + tl.sum(weights, axis=1)
        out_acc = out_acc * alpha[:, None]

        v_ptrs = (
            v_ptr
            + batch * stride_vb
            + kv_head * stride_vh
            + k_offs[:, None] * stride_vn
            + d_offs[None, :] * stride_vd
        )
        value = tl.load(v_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0)
        out_acc = out_acc + tl.dot(weights.to(value.dtype), value)

        running_m = new_m

    denom = running_l[:, None]
    out_tile = tl.where(denom > 0, out_acc / denom, 0.0)

    o_ptrs = (
        o_ptr
        + batch * stride_ob
        + q_head * stride_oh
        + q_offs[:, None] * stride_om
        + d_offs[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        out_tile.to(o_ptr.dtype.element_ty),
        mask=q_valid[:, None] & d_valid[None, :],
    )


def launch_tiled_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    scale: float,
    causal: bool,
    block_q: int = 64,
    block_kv: int = 64,
) -> None:
    """Launch the CUDA Triton kernel into ``out``."""
    batch, n_q_heads, seq_q, head_dim = q.shape
    n_kv_heads = k.shape[1]
    seq_k = k.shape[2]
    if head_dim < TRITON_MIN_HEAD_DIM or head_dim > TRITON_MAX_HEAD_DIM:
        raise ValueError(
            f"Triton path supports head_dim in [{TRITON_MIN_HEAD_DIM}, "
            f"{TRITON_MAX_HEAD_DIM}]; got {head_dim}. Use the labeled "
            "torch_reference path for other widths."
        )
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"query heads ({n_q_heads}) must be a multiple of kv heads ({n_kv_heads})"
        )

    block_dim = max(16, _next_pow2(head_dim))
    has_mask = attn_mask is not None
    mask_is_bool = bool(has_mask and attn_mask.dtype == torch.bool)
    if has_mask:
        mask_view = broadcast_attn_mask(attn_mask, batch, n_q_heads, seq_q, seq_k)
        mask_ptr = mask_view
        stride_mb, stride_mh, stride_mm, stride_mn = mask_view.stride()
    else:
        mask_ptr = q
        stride_mb = stride_mh = stride_mm = stride_mn = 0

    grid = (triton.cdiv(seq_q, block_q), batch * n_q_heads)
    _szl_tiled_attn_kernel[grid](
        q,
        k,
        v,
        out,
        mask_ptr,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        seq_q,
        seq_k,
        n_q_heads,
        n_kv_heads,
        head_dim,
        float(scale),
        HAS_MASK=has_mask,
        MASK_IS_BOOL=mask_is_bool,
        CAUSAL=causal,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_DIM=block_dim,
        num_warps=4,
        num_stages=2,
    )
