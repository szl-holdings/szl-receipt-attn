# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (szl-receipt-attn).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
CPU-only, deterministic and fast: the torch-reference path against torch SDPA, plus one
receipt on a chain that must verify. The result's repr is written to the receipt.
"""
from __future__ import annotations


def smoke(kernel):
    import torch
    import torch.nn.functional as F

    check = kernel.selfcheck()
    if not check.get("ok"):
        raise AssertionError(f"selfcheck() did not pass: {check}")

    gen = torch.Generator().manual_seed(20261013)
    q, k, v = (torch.randn(1, 2, 16, 32, generator=gen) for _ in range(3))
    chain = kernel.ReceiptChain()
    y = kernel.receipt_attn(q, k, v, causal=True, chain=chain)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    err = float((y - ref).abs().max().item())
    if err > 1e-5:
        raise AssertionError(f"receipt_attn differs from SDPA on CPU: max abs {err}")
    ok, depth, brk = chain.verify()
    if not (ok and depth == 1):
        raise AssertionError(f"receipt chain did not verify: ok={ok} depth={depth} break={brk}")
    return {"selfcheck_ok": True, "max_abs_vs_sdpa": err, "chain_depth": depth, "path": "torch_reference"}
