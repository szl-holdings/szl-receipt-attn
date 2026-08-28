# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests vs in-repo SDPA equivalent and torch SDPA.

Documented tolerances (see CARD.md / README.md):

* float32: atol=1e-4, rtol=1e-4
* float16: atol=2e-2, rtol=1e-2
* bfloat16: atol=2e-2, rtol=1.6e-2

CUDA/Triton tests skip cleanly when no GPU is present. A skip is not a pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

TOLERANCES = {
    torch.float32: (1.0e-4, 1.0e-4),
    torch.float16: (2.0e-2, 1.0e-2),
    torch.bfloat16: (2.0e-2, 1.6e-2),
}


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _qkv(device, dtype, batch=2, heads=4, seq=17, dim=32, kv_heads=None):
    torch.manual_seed(2026)
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.kernels_ci
@pytest.mark.parametrize("seq", [17, 64])
@pytest.mark.parametrize("causal", [False, True])
def test_cpu_tiled_matches_sdpa_equivalent(kernel_mod, cpu_device, seq, causal):
    q, k, v = _qkv(cpu_device, torch.float32, seq=seq)
    got = kernel_mod.receipt_attn(q, k, v, causal=causal)
    ref = kernel_mod.sdpa_equivalent(q, k, v, causal=causal)
    _assert_close(got, ref, torch.float32)


@pytest.mark.kernels_ci
def test_cpu_matches_torch_sdpa(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=32)
    got = kernel_mod.receipt_attn(q, k, v, causal=True)
    torch_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _assert_close(got, torch_sdpa, torch.float32)


@pytest.mark.kernels_ci
def test_bool_attn_mask(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=24)
    mask = torch.ones(24, 24, dtype=torch.bool, device=cpu_device)
    mask[:, 20:] = False
    got = kernel_mod.receipt_attn(q, k, v, attn_mask=mask)
    ref = kernel_mod.sdpa_equivalent(q, k, v, attn_mask=mask)
    _assert_close(got, ref, torch.float32)


@pytest.mark.kernels_ci
def test_additive_attn_mask(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=16)
    additive = torch.zeros(16, 16, device=cpu_device, dtype=torch.float32)
    additive[:, 8:] = float("-inf")
    got = kernel_mod.receipt_attn(q, k, v, attn_mask=additive)
    ref = kernel_mod.sdpa_equivalent(q, k, v, attn_mask=additive)
    _assert_close(got, ref, torch.float32)


@pytest.mark.kernels_ci
def test_grouped_query_attention(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, heads=8, kv_heads=2, seq=20, dim=32)
    got = kernel_mod.receipt_attn(q, k, v, causal=True)
    ref = kernel_mod.sdpa_equivalent(q, k, v, causal=True)
    _assert_close(got, ref, torch.float32)


@pytest.mark.kernels_ci
def test_receipt_chain_records_and_verifies(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32)
    chain = kernel_mod.UnifiedReceiptChain()
    out = kernel_mod.receipt_attn(q, k, v, causal=True, chain=chain)
    assert out.shape == q.shape
    ok, depth, first_break = chain.verify()
    assert ok is True
    assert depth == 1
    assert first_break is None
    body = chain.receipts()[0]["body"]
    assert body["path"] == kernel_mod.PATH_TORCH_REFERENCE
    assert body["causal"] is True
    assert body["mask_digest"] == "none"


@pytest.mark.kernels_ci
def test_receipt_tamper_stays_failed(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=8, dim=32)
    chain = kernel_mod.UnifiedReceiptChain()
    kernel_mod.receipt_attn(q, k, v, chain=chain)
    tampered = kernel_mod.UnifiedReceiptChain.from_json(chain.to_json())
    tampered._receipts[0]["body"]["q_shape"] = [0]
    ok, _, first_break = tampered.verify()
    assert ok is False
    assert first_break == 0


@pytest.mark.kernels_ci
def test_selfcheck_never_fabricates_pass(kernel_mod):
    report = kernel_mod.selfcheck()
    assert report["fabricated"] is False
    assert report["lambda"].startswith("Conjecture 1")
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    if not torch.cuda.is_available():
        assert statuses["triton_cuda_vs_sdpa"] == "skipped"
    for check in report["checks"]:
        assert check["status"] in {"pass", "fail", "skipped"}
        if check["status"] == "skipped":
            assert "not a pass" in check["detail"].lower() or "skip" in check["detail"].lower() or "unavailable" in check["detail"].lower() or "not exercised" in check["detail"].lower()
    if any(c["status"] == "fail" for c in report["checks"]):
        assert report["ok"] is False


@pytest.mark.kernels_ci
def test_cuda_triton_matches_sdpa_or_skips(kernel_mod, cuda_device):
    if not kernel_mod.triton_cuda_available(cuda_device):
        pytest.skip("Triton not importable on this CUDA device (honest skip, not a pass)")
    q, k, v = _qkv(cuda_device, torch.float16, seq=64, dim=64)
    got = kernel_mod.receipt_attn(q, k, v, causal=True)
    ref = kernel_mod.sdpa_equivalent(q, k, v, causal=True)
    _assert_close(got, ref, torch.float16)
    chain = kernel_mod.UnifiedReceiptChain()
    kernel_mod.receipt_attn(q, k, v, causal=True, chain=chain)
    assert chain.receipts()[0]["body"]["path"] == kernel_mod.PATH_TRITON_CUDA
