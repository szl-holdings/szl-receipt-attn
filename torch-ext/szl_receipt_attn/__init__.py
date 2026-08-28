# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""szl-receipt-attn — original receipt-aware tiled fused attention."""

from __future__ import annotations

from .api import receipt_attn, triton_cuda_available
from .const import (
    DOCTRINE,
    HOMEPAGE,
    LAMBDA_STATUS,
    OWNER,
    PATH_LABELS,
    PATH_TORCH_REFERENCE,
    PATH_TRITON_CUDA,
    TOLERANCES,
)
from .receipt import UnifiedReceiptChain, digest_mask
from .reference import sdpa_equivalent, tiled_online_softmax_attn
from .selfcheck import selfcheck

# Register namespaced Torch ops when kernel-builder generated `_ops`.
# ``add_op_namespace_prefix`` itself is imported only in ops.py, with
# no fallback (kernel-builder static analysis).
try:
    from .ops import _receipt_attn_fwd as _receipt_attn_fwd
except ImportError:
    _receipt_attn_fwd = None

__all__ = [
    "DOCTRINE",
    "HOMEPAGE",
    "LAMBDA_STATUS",
    "OWNER",
    "PATH_LABELS",
    "PATH_TORCH_REFERENCE",
    "PATH_TRITON_CUDA",
    "TOLERANCES",
    "UnifiedReceiptChain",
    "digest_mask",
    "receipt_attn",
    "sdpa_equivalent",
    "selfcheck",
    "tiled_online_softmax_attn",
    "triton_cuda_available",
]
