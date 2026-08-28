# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Namespaced Torch custom op registration for kernel-builder.

``add_op_namespace_prefix`` is imported from the generated ``_ops``
module with no fallback. This file is imported only when ``_ops`` exists
(Hub / kernel-builder builds).
"""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix
from .triton_kernel import launch_tiled_attn


@torch.library.custom_op(add_op_namespace_prefix("receipt_attn_fwd"), mutates_args={"out"})
def _receipt_attn_fwd(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    causal: bool,
    has_mask: bool,
) -> None:
    mask: Optional[torch.Tensor] = attn_mask if has_mask else None
    launch_tiled_attn(q, k, v, out, attn_mask=mask, scale=scale, causal=causal)


@_receipt_attn_fwd.register_fake
def _(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    causal: bool,
    has_mask: bool,
) -> None:
    return None
