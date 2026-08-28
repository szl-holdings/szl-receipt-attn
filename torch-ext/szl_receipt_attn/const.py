# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Honesty-labeled constants for szl-receipt-attn."""

from __future__ import annotations

from typing import Mapping

PACKAGE_NAME = "szl-receipt-attn"
DOCTRINE = "v11 LOCKED"
LAMBDA_STATUS = (
    "Conjecture 1 (OPEN) — advisory, uniqueness unproven, never proven trust"
)
HOMEPAGE = "https://a-11-oy.com"
OWNER = "Stephen P. Lutar Jr. / SZL Holdings"

# Path labels. CARD/README must match these strings.
PATH_TRITON_CUDA = "triton_cuda"
PATH_TORCH_REFERENCE = "torch_reference"

PATH_LABELS: Mapping[str, str] = {
    PATH_TRITON_CUDA: (
        "Triton JIT tiled fused-attention kernel. Runs on CUDA only. "
        "Not a CPU kernel. No speedup is claimed."
    ),
    PATH_TORCH_REFERENCE: (
        "Pure-PyTorch tiled online-softmax reference. Used for CPU tests "
        "and as the in-repo SDPA silhouette. Labeled reference, not a "
        "compiled CPU backend."
    ),
}

# Documented comparison tolerances vs the in-repo SDPA equivalent / torch SDPA.
# These are correctness bands, not quality or speed claims.
TOLERANCES: Mapping[str, Mapping[str, float]] = {
    "float32": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "float16": {"atol": 2.0e-2, "rtol": 1.0e-2},
    "bfloat16": {"atol": 2.0e-2, "rtol": 1.6e-2},
}

TRITON_MAX_HEAD_DIM = 128
TRITON_MIN_HEAD_DIM = 16
DEFAULT_BLOCK_Q = 64
DEFAULT_BLOCK_KV = 64
