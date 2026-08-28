# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Torch reference + dispatcher. Original SZL cut. No Dao source.

Named attn.py (not _ops.py): kernel-builder generates
torch-ext/<name>/_ops.py with add_op_namespace_prefix. KERNEL original
dispatcher is unchanged here.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ._chain import ReceiptChain


def _mask_digest(attn_mask: Optional[torch.Tensor], causal: bool) -> str:
    bits = f"causal={int(causal)}"
    if attn_mask is None:
        return bits + ":none"
    h = float(torch.sum(attn_mask.to(torch.float32)).item())
    return f"{bits}:sum={h:.6g}:shape={tuple(attn_mask.shape)}"


def _torch_attn(q, k, v, *, causal, attn_mask, scale):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=scale
    )


def receipt_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    chain: Optional[ReceiptChain] = None,
    scale: Optional[float] = None,
    prefer: str = "auto",
) -> torch.Tensor:
    path = "torch_reference"
    use_triton = prefer == "triton" or (
        prefer == "auto" and q.is_cuda and attn_mask is None and q.shape[-1] <= 32
    )
    if use_triton:
        try:
            from ._triton import triton_attn

            y = triton_attn(q, k, v, causal=causal, scale=scale)
            path = "triton"
        except Exception:
            y = _torch_attn(q, k, v, causal=causal, attn_mask=attn_mask, scale=scale)
            path = "torch_reference_fallback"
    else:
        y = _torch_attn(q, k, v, causal=causal, attn_mask=attn_mask, scale=scale)
    if chain is not None:
        chain.emit(
            {
                "op": "receipt_attn",
                "path": path,
                "q_shape": list(q.shape),
                "dtype": str(q.dtype).replace("torch.", ""),
                "causal": causal,
                "mask": _mask_digest(attn_mask, causal),
                "lambda": "Conjecture 1",
            }
        )
    return y


def selfcheck() -> dict:
    torch.manual_seed(20260828)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    chain = ReceiptChain()
    y = receipt_attn(q, k, v, causal=True, chain=chain, prefer="torch")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    err = float((y - ref).abs().max().item())
    ok_chain, depth, brk = chain.verify()
    ok = bool(err < 1e-5 and ok_chain and depth == 1)
    return {
        "ok": ok,
        "max_abs_vs_sdpa": err,
        "chain_ok": ok_chain,
        "chain_depth": depth,
        "chain_break": brk,
        "path": "torch_reference",
        "lambda": "Conjecture 1",
        "note": "correctness vs torch SDPA only; no speedup claimed",
    }
