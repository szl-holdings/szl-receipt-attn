# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Torch reference + dispatcher. Original SZL cut. No Dao source.

Named attn.py (not _ops.py): kernel-builder generates
torch-ext/<name>/_ops.py with add_op_namespace_prefix. KERNEL original
dispatcher is unchanged here.
"""
from __future__ import annotations

import hashlib
import sys
from typing import Optional

import torch
import torch.nn.functional as F

from ._chain import ReceiptChain

# Version tag carried in-band by every receipt "mask" value. The earlier
# unversioned form ("causal=<b>:none" or "causal=<b>:sum=<total>:shape=(...)")
# bound only the element sum, so equal-count masks with different topology
# collided. Nothing in the estate parses that form and it is not emitted any
# more; old chains still verify because ReceiptChain hashes the value opaquely.
# An unversioned value can never equal a v1 value, which always starts with
# this tag, so readers can tell the two apart and reject what they don't know.
MASK_IDENTITY_VERSION = "szl-mask-id/v1"

_HASH_CHUNK_ELEMS = 1 << 20


def _mask_digest(attn_mask: Optional[torch.Tensor], causal: bool) -> str:
    """Versioned canonical identity of ``(attn_mask, causal)`` for receipts.

    ``szl-mask-id/v1;causal=<0|1>;mask=none`` when there is no mask, else
    ``szl-mask-id/v1;causal=<0|1>;dtype=<dtype>;shape=<d0>x<d1>...;sha256=<hex>``.
    The digest is SHA-256 over the mask's canonical bytes: elements in C
    (row-major) order of the logical shape, so memory layout, strides and
    device do not matter; bool masks as one uint8 0/1 per element; every
    other dtype as its own element bytes, little-endian. dtype stays in the
    identity because SDPA reads bool masks as keep-masks and float masks as
    additive biases. The identity binds exact bytes, never semantic
    equivalence (e.g. -0.0 and 0.0 bias entries get different identities).
    """
    head = f"{MASK_IDENTITY_VERSION};causal={int(bool(causal))}"
    if attn_mask is None:
        return f"{head};mask=none"
    canonical = attn_mask.detach().to(device="cpu")
    dtype = str(canonical.dtype).replace("torch.", "")
    if canonical.dtype == torch.bool:
        canonical = canonical.to(torch.uint8)
    shape = "x".join(str(int(dim)) for dim in canonical.shape)
    flat = canonical.contiguous().reshape(-1)
    width = flat.element_size()
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), _HASH_CHUNK_ELEMS):
        raw = flat[start : start + _HASH_CHUNK_ELEMS].contiguous().view(torch.uint8)
        if width > 1 and sys.byteorder == "big":
            raw = raw.reshape(-1, width).flip(-1).reshape(-1)
        digest.update(bytes(raw.tolist()))
    return f"{head};dtype={dtype};shape={shape};sha256={digest.hexdigest()}"


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
    # The Triton kernel has no arbitrary-mask input. Preserve mask semantics
    # even when callers prefer Triton by selecting the reference backend.
    use_triton = attn_mask is None and (
        prefer == "triton" or (prefer == "auto" and q.is_cuda and q.shape[-1] <= 32)
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
