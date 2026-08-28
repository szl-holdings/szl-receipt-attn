# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Public API: receipt_attn, path selection, optional receipt emit."""

from __future__ import annotations

from typing import Optional

import torch

from .const import (
    PATH_TORCH_REFERENCE,
    PATH_TRITON_CUDA,
    TRITON_MAX_HEAD_DIM,
    TRITON_MIN_HEAD_DIM,
)
from .receipt import UnifiedReceiptChain, digest_mask, shape_list
from .reference import tiled_online_softmax_attn

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def triton_cuda_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, v must be 4D in layout (batch, heads, seq, head_dim); "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share a dtype")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {q.dtype}; expected {_SUPPORTED_DTYPES}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions of q, k, v must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of heads")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v head_dim must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"query heads ({q.shape[1]}) must be a multiple of kv heads ({k.shape[1]})"
        )
    if causal and q.shape[2] != k.shape[2]:
        raise ValueError(
            "causal=True requires equal query and key sequence lengths; "
            "use attn_mask for unequal lengths"
        )


def _select_path(
    q: torch.Tensor, *, prefer_triton: bool
) -> str:
    head_dim = int(q.shape[-1])
    triton_ok = (
        prefer_triton
        and triton_cuda_available(q.device)
        and TRITON_MIN_HEAD_DIM <= head_dim <= TRITON_MAX_HEAD_DIM
    )
    if triton_ok:
        return PATH_TRITON_CUDA
    return PATH_TORCH_REFERENCE


def receipt_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    chain: Optional[UnifiedReceiptChain] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Receipt-aware tiled fused attention.

    Layout: ``(batch, heads, seq, head_dim)``. Grouped-query attention is
    supported when ``q.shape[1]`` is a multiple of ``k.shape[1]``.

    Paths (honesty):

    * **triton_cuda** — Triton JIT kernel, CUDA only.
    * **torch_reference** — pure PyTorch tiled online softmax (CPU tests
      and any case where Triton cannot run).

    No speedup is claimed. Optional ``chain`` records a SHA3-256 receipt
    of shapes, dtype, causal flag, and mask digest. Λ is advisory.
    """
    _validate(q, k, v, causal=causal)
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5
    else:
        scale = float(scale)

    path = _select_path(q, prefer_triton=True)
    if path == PATH_TRITON_CUDA:
        from .triton_kernel import launch_tiled_attn

        out = torch.empty_like(q)
        launch_tiled_attn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            attn_mask=attn_mask,
            scale=scale,
            causal=causal,
        )
    else:
        out = tiled_online_softmax_attn(
            q, k, v, causal=causal, attn_mask=attn_mask, scale=scale
        )

    if chain is not None:
        chain.emit(
            {
                "causal": bool(causal),
                "dtype": str(q.dtype).replace("torch.", ""),
                "k_shape": shape_list(k),
                "mask_digest": digest_mask(attn_mask),
                "op": "receipt_attn",
                "path": path,
                "q_shape": shape_list(q),
                "scale": scale,
                "v_shape": shape_list(v),
            }
        )
    return out
