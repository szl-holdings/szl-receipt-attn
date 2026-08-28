# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch tiled online-softmax attention (CPU/CUDA reference path).

This is the labeled reference silhouette for szl-receipt-attn:

* ``sdpa_equivalent`` — one-shot matmul + softmax (the comparison target).
* ``tiled_online_softmax_attn`` — the same online-softmax recurrence the
  Triton kernel implements, written in PyTorch so CPU tests can run
  without a GPU.

Algorithm (papers, not vendored kernels):

* Tile Q along the query sequence and (K, V) along the key sequence.
* Keep running row-max ``m``, running sum ``l``, and output accumulator
  ``O`` (Milakov & Gimelshein online softmax; Dao et al. FA1/FA2/FA3).
* For each KV tile: ``m' = max(m, rowmax(S))``,
  ``P = exp(S - m')``,
  ``l' = exp(m - m') * l + rowsum(P)``,
  ``O' = exp(m - m') * O + P @ V``.
* Finish with ``O / l``. Fully-masked rows return zeros (defined),
  whereas one-shot softmax of all ``-inf`` may yield NaN.

This file is an original SZL construction. It is not copied from
Dao-AILab/flash-attention or kernels-community flash-attn packages.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .const import DEFAULT_BLOCK_KV, DEFAULT_BLOCK_Q


def expand_kv_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repeat KV heads for grouped-query attention when needed."""
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads == kv_heads:
        return q, k, v
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"query heads ({q_heads}) must be a multiple of kv heads ({kv_heads})"
        )
    repeats = q_heads // kv_heads
    return q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)


def broadcast_attn_mask(
    attn_mask: torch.Tensor, batch: int, heads: int, seq_q: int, seq_k: int
) -> torch.Tensor:
    if attn_mask.ndim == 2:
        mask = attn_mask[None, None, :, :]
    elif attn_mask.ndim == 3:
        mask = attn_mask[:, None, :, :]
    elif attn_mask.ndim == 4:
        mask = attn_mask
    else:
        raise ValueError(
            "attn_mask must be 2D (S,S), 3D (B,S,S), or 4D (B,H,S,S); "
            f"got ndim={attn_mask.ndim}"
        )
    if mask.shape[-2] != seq_q or mask.shape[-1] != seq_k:
        raise ValueError(
            f"attn_mask trailing dims {tuple(mask.shape[-2:])} != {(seq_q, seq_k)}"
        )
    try:
        return mask.expand(batch, heads, seq_q, seq_k)
    except RuntimeError as exc:
        raise ValueError(
            f"attn_mask shape {tuple(attn_mask.shape)} does not broadcast to "
            f"{(batch, heads, seq_q, seq_k)}"
        ) from exc


def _apply_score_masks(
    scores: torch.Tensor,
    *,
    causal: bool,
    attn_mask: Optional[torch.Tensor],
    q_start: int,
    k_start: int,
) -> torch.Tensor:
    seq_q = scores.shape[-2]
    seq_k = scores.shape[-1]
    if causal:
        q_idx = torch.arange(q_start, q_start + seq_q, device=scores.device)[:, None]
        k_idx = torch.arange(k_start, k_start + seq_k, device=scores.device)[None, :]
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
    if attn_mask is None:
        return scores
    tile = attn_mask[..., q_start : q_start + seq_q, k_start : k_start + seq_k]
    if tile.dtype == torch.bool:
        return scores.masked_fill(~tile, float("-inf"))
    return scores + tile.to(dtype=scores.dtype)


def sdpa_equivalent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """One-shot scaled dot-product attention (matmul + softmax).

    In-repo comparison target. Not FlashAttention. Numerics are fp32
    internally, then cast back to the input dtype.
    """
    q, k, v = expand_kv_heads(q, k, v)
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    if scale is None:
        scale = head_dim ** -0.5
    mask = (
        broadcast_attn_mask(attn_mask, batch, heads, seq_q, seq_k)
        if attn_mask is not None
        else None
    )
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
    scores = _apply_score_masks(scores, causal=causal, attn_mask=mask, q_start=0, k_start=0)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float())
    return out.to(dtype=q.dtype)


def tiled_online_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> torch.Tensor:
    """Tiled online-softmax attention in PyTorch (labeled reference path)."""
    q, k, v = expand_kv_heads(q, k, v)
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    if scale is None:
        scale = head_dim ** -0.5
    scale_f = float(scale)
    mask = (
        broadcast_attn_mask(attn_mask, batch, heads, seq_q, seq_k)
        if attn_mask is not None
        else None
    )

    qf = q.float()
    kf = k.float()
    vf = v.float()
    out = torch.zeros(batch, heads, seq_q, head_dim, dtype=torch.float32, device=q.device)

    for q0 in range(0, seq_q, block_q):
        q1 = min(q0 + block_q, seq_q)
        q_tile = qf[:, :, q0:q1, :]
        bq = q1 - q0
        running_m = torch.full(
            (batch, heads, bq), float("-inf"), dtype=torch.float32, device=q.device
        )
        running_l = torch.zeros(batch, heads, bq, dtype=torch.float32, device=q.device)
        acc = torch.zeros(
            batch, heads, bq, head_dim, dtype=torch.float32, device=q.device
        )

        for k0 in range(0, seq_k, block_kv):
            k1 = min(k0 + block_kv, seq_k)
            k_tile = kf[:, :, k0:k1, :]
            v_tile = vf[:, :, k0:k1, :]
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale_f
            scores = _apply_score_masks(
                scores, causal=causal, attn_mask=mask, q_start=q0, k_start=k0
            )

            tile_max = scores.amax(dim=-1)
            new_m = torch.maximum(running_m, tile_max)
            finite = torch.isfinite(new_m)
            alpha = torch.exp(running_m - new_m)
            alpha = torch.where(finite, alpha, torch.zeros_like(alpha))

            weights = torch.exp(scores - new_m.unsqueeze(-1))
            weights = torch.where(finite.unsqueeze(-1), weights, torch.zeros_like(weights))
            weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

            running_l = running_l * alpha + weights.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(weights, v_tile)
            running_m = new_m

        denom = running_l.unsqueeze(-1)
        tile_out = torch.where(denom > 0, acc / denom, torch.zeros_like(acc))
        out[:, :, q0:q1, :] = tile_out

    return out.to(dtype=q.dtype)
