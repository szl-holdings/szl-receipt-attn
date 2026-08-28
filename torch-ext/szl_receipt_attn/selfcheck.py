# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Honesty-labeled self-check. Never fabricates a pass."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .api import receipt_attn, triton_cuda_available
from .const import (
    DOCTRINE,
    LAMBDA_STATUS,
    PACKAGE_NAME,
    PATH_LABELS,
    PATH_TORCH_REFERENCE,
    PATH_TRITON_CUDA,
    TOLERANCES,
)
from .receipt import UnifiedReceiptChain
from .reference import sdpa_equivalent, tiled_online_softmax_attn


def _tol_for(dtype: torch.dtype) -> tuple[float, float]:
    key = str(dtype).replace("torch.", "")
    band = TOLERANCES[key]
    return float(band["atol"]), float(band["rtol"])


def _status(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    record = {"name": name, "status": status, "detail": detail}
    record.update(extra)
    return record


def _compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return _status(
            name,
            "fail",
            f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}",
        )
    finite_actual = torch.isfinite(actual)
    finite_expected = torch.isfinite(expected)
    if not torch.equal(finite_actual, finite_expected):
        return _status(name, "fail", "finite-mask mismatch vs reference")
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1.0)
    rel = diff / denom
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    ok = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    return _status(
        name,
        "pass" if ok else "fail",
        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e} atol={atol} rtol={rtol}",
        max_abs=max_abs,
        max_rel=max_rel,
        atol=atol,
        rtol=rtol,
    )


def _make_inputs(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    q = torch.randn(2, 4, 17, 32, device=device, dtype=dtype)
    k = torch.randn(2, 4, 17, 32, device=device, dtype=dtype)
    v = torch.randn(2, 4, 17, 32, device=device, dtype=dtype)
    return q, k, v


def selfcheck() -> dict[str, Any]:
    """Run honesty-labeled checks. Inspect the returned dict.

    ``ok`` is True only when every *executed* check passed. Skipped
    CUDA/Triton checks are ``skipped``, never ``pass``. A failed check
    stays failed. This function does not fabricate a green result.
    """
    checks: list[dict[str, Any]] = []
    cpu = torch.device("cpu")
    dtype = torch.float32
    atol, rtol = _tol_for(dtype)
    q, k, v = _make_inputs(cpu, dtype)

    tiled = tiled_online_softmax_attn(q, k, v, causal=True)
    equivalent = sdpa_equivalent(q, k, v, causal=True)
    checks.append(
        _compare(
            "cpu_tiled_vs_sdpa_equivalent",
            tiled,
            equivalent,
            atol=atol,
            rtol=rtol,
        )
    )

    try:
        torch_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        checks.append(
            _compare(
                "cpu_tiled_vs_torch_sdpa",
                tiled,
                torch_sdpa,
                atol=atol,
                rtol=rtol,
            )
        )
    except Exception as exc:  # pragma: no cover - backend may be absent
        checks.append(
            _status(
                "cpu_tiled_vs_torch_sdpa",
                "skipped",
                f"torch SDPA unavailable: {type(exc).__name__}: {exc}",
            )
        )

    public = receipt_attn(q, k, v, causal=True)
    checks.append(
        _compare(
            "cpu_receipt_attn_uses_torch_reference",
            public,
            tiled,
            atol=0.0,
            rtol=0.0,
        )
    )
    if public.device.type == "cpu":
        checks.append(
            _status(
                "cpu_path_label",
                "pass",
                PATH_LABELS[PATH_TORCH_REFERENCE],
                path=PATH_TORCH_REFERENCE,
            )
        )
    else:
        checks.append(
            _status(
                "cpu_path_label",
                "fail",
                f"expected {PATH_TORCH_REFERENCE} on CPU, got device {public.device}",
            )
        )

    chain = UnifiedReceiptChain()
    _ = receipt_attn(q, k, v, causal=True, chain=chain)
    ok, depth, first_break = chain.verify()
    checks.append(
        _status(
            "receipt_chain_verify",
            "pass" if ok and depth == 1 else "fail",
            f"ok={ok} depth={depth} first_break={first_break}",
            ok=ok,
            depth=depth,
            first_break=first_break,
        )
    )

    tampered = UnifiedReceiptChain.from_json(chain.to_json())
    if tampered._receipts:
        tampered._receipts[0]["body"]["causal"] = not tampered._receipts[0]["body"]["causal"]
    tamper_ok, _, break_at = tampered.verify()
    checks.append(
        _status(
            "receipt_chain_tamper_stays_failed",
            "pass" if (not tamper_ok) and break_at is not None else "fail",
            f"tampered verify ok={tamper_ok} first_break={break_at} "
            "(a failed check must stay failed)",
            tampered_ok=tamper_ok,
            first_break=break_at,
        )
    )

    cuda = torch.device("cuda") if torch.cuda.is_available() else None
    if cuda is None:
        checks.append(
            _status(
                "triton_cuda_vs_sdpa",
                "skipped",
                "No CUDA GPU. Triton path not exercised. This is not a pass.",
                path=PATH_TRITON_CUDA,
            )
        )
    elif not triton_cuda_available(cuda):
        checks.append(
            _status(
                "triton_cuda_vs_sdpa",
                "skipped",
                "CUDA present but Triton is not importable. Not a pass.",
                path=PATH_TRITON_CUDA,
            )
        )
    else:
        try:
            q_c, k_c, v_c = _make_inputs(cuda, torch.float16)
            atol16, rtol16 = _tol_for(torch.float16)
            got = receipt_attn(q_c, k_c, v_c, causal=True)
            ref = sdpa_equivalent(q_c, k_c, v_c, causal=True)
            cmp16 = _compare(
                "triton_cuda_vs_sdpa",
                got,
                ref,
                atol=atol16,
                rtol=rtol16,
            )
            cmp16["path"] = PATH_TRITON_CUDA
            cmp16["device"] = str(got.device)
            checks.append(cmp16)
        except Exception as exc:
            checks.append(
                _status(
                    "triton_cuda_vs_sdpa",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                    path=PATH_TRITON_CUDA,
                )
            )

    executed = [c for c in checks if c["status"] != "skipped"]
    failed = [c for c in executed if c["status"] != "pass"]
    report: dict[str, Any] = {
        "kernel": PACKAGE_NAME,
        "doctrine": DOCTRINE,
        "lambda": LAMBDA_STATUS,
        "fabricated": False,
        "ok": len(executed) > 0 and len(failed) == 0,
        "executed": len(executed),
        "failed": len(failed),
        "skipped": sum(1 for c in checks if c["status"] == "skipped"),
        "checks": checks,
        "paths": dict(PATH_LABELS),
        "tolerances": {k: dict(v) for k, v in TOLERANCES.items()},
        "note": (
            "ok means every executed check passed. skipped CUDA/Triton "
            "checks are not passes. No speedup or benchmark is claimed."
        ),
    }
    return report
