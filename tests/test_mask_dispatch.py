# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""RQ-09: arbitrary masks must not reach a backend without mask support.

CPU tests substitute a successful unmasked backend, so they detect the
dispatcher bug even when Triton is absent. CUDA tests are separate and
require a real successful backend; a fallback cannot count as a GPU pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
from szl_receipt_attn import ReceiptChain, receipt_attn  # noqa: E402


def _inputs(device="cpu"):
    q = torch.zeros(1, 1, 2, 16, device=device)
    k = torch.zeros_like(q)
    v = torch.empty_like(q)
    v[:, :, 0, :] = 1.0
    v[:, :, 1, :] = 7.0
    return q, k, v


def _masks(kind, device="cpu"):
    if kind == "bool":
        first = torch.tensor([[True, False], [True, False]], device=device)
    else:
        first = torch.tensor([[0.0, float("-inf")], [0.0, float("-inf")]], device=device)
    return first, first.flip(-1)


def _fake_successful_triton(monkeypatch):
    calls = []
    fake = ModuleType("szl_receipt_attn._triton")

    def unmasked(q, k, v, *, causal, scale):
        calls.append((causal, scale))
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=scale, dropout_p=0.0
        )

    fake.triton_attn = unmasked
    monkeypatch.setitem(sys.modules, "szl_receipt_attn._triton", fake)
    return calls


@pytest.mark.parametrize("prefer", ["auto", "torch", "triton"])
@pytest.mark.parametrize("kind", ["bool", "float"])
def test_mask_preserved_with_successful_unmasked_backend(monkeypatch, prefer, kind):
    calls = _fake_successful_triton(monkeypatch)
    q, k, v = _inputs()
    chain = ReceiptChain()
    outputs = []
    for mask in _masks(kind):
        got = receipt_attn(q, k, v, attn_mask=mask, prefer=prefer, scale=0.25, chain=chain)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.25)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        outputs.append(got)
    assert not calls, "A masked call reached a backend that cannot accept masks"
    assert not torch.equal(outputs[0], outputs[1]), "Mask topology must affect this fixture"
    assert [row["path"] for row in chain._rows] == ["torch_reference", "torch_reference"]
    assert chain._rows[0]["mask"] != chain._rows[1]["mask"]
    assert chain.verify() == (True, 2, -1)


def test_unmasked_forced_backend_still_runs(monkeypatch):
    calls = _fake_successful_triton(monkeypatch)
    q, k, v = _inputs()
    chain = ReceiptChain()
    got = receipt_attn(q, k, v, prefer="triton", chain=chain, scale=0.25)
    expected = F.scaled_dot_product_attention(q, k, v, scale=0.25)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    assert calls == [(False, 0.25)]
    assert chain._rows[0]["path"] == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU; not a GPU pass")
@pytest.mark.parametrize("kind", ["bool", "float"])
def test_real_cuda_masked_forced_preference_routes_reference(kind):
    q, k, v = _inputs("cuda")
    chain = ReceiptChain()
    outputs = []
    for mask in _masks(kind, "cuda"):
        got = receipt_attn(q, k, v, attn_mask=mask, prefer="triton", chain=chain)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        outputs.append(got)
    assert not torch.equal(outputs[0], outputs[1])
    assert [row["path"] for row in chain._rows] == ["torch_reference", "torch_reference"]
    assert chain.verify() == (True, 2, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU; not a Triton pass")
def test_real_cuda_unmasked_backend_cannot_hide_fallback():
    pytest.importorskip("triton", reason="No Triton installed; not a Triton pass")
    q, k, v = _inputs("cuda")
    chain = ReceiptChain()
    got = receipt_attn(q, k, v, prefer="triton", chain=chain)
    assert chain._rows[0]["path"] == "triton", "A fallback cannot qualify the Triton path"
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
