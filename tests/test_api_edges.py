# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Extra correctness tests. No benchmarks. CUDA tests skip if no GPU."""

from __future__ import annotations

import pytest
import torch

# Same documented bands as tests/test_receipt_attn.py (fp32 CPU edges).
_FP32 = (1.0e-4, 1.0e-4)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = _FP32
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _qkv(device, dtype, batch=2, heads=4, seq=17, dim=32, kv_heads=None):
    torch.manual_seed(2026)
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.kernels_ci
def test_custom_scale(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=16, dim=32)
    scale = 0.25
    got = kernel_mod.receipt_attn(q, k, v, scale=scale)
    ref = kernel_mod.sdpa_equivalent(q, k, v, scale=scale)
    _assert_close(got, ref)


@pytest.mark.kernels_ci
def test_unequal_seq_non_causal(kernel_mod, cpu_device):
    torch.manual_seed(11)
    q = torch.randn(1, 2, 12, 32, device=cpu_device)
    k = torch.randn(1, 2, 20, 32, device=cpu_device)
    v = torch.randn(1, 2, 20, 32, device=cpu_device)
    got = kernel_mod.receipt_attn(q, k, v, causal=False)
    ref = kernel_mod.sdpa_equivalent(q, k, v, causal=False)
    _assert_close(got, ref)


@pytest.mark.kernels_ci
def test_causal_unequal_seq_raises(kernel_mod, cpu_device):
    q = torch.randn(1, 2, 8, 32, device=cpu_device)
    k = torch.randn(1, 2, 12, 32, device=cpu_device)
    v = torch.randn(1, 2, 12, 32, device=cpu_device)
    with pytest.raises(ValueError, match="causal=True requires equal"):
        kernel_mod.receipt_attn(q, k, v, causal=True)


@pytest.mark.kernels_ci
def test_four_d_bool_mask(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, batch=2, heads=4, seq=16, dim=32)
    mask = torch.ones(2, 4, 16, 16, dtype=torch.bool, device=cpu_device)
    mask[:, :, :, 12:] = False
    got = kernel_mod.receipt_attn(q, k, v, attn_mask=mask)
    ref = kernel_mod.sdpa_equivalent(q, k, v, attn_mask=mask)
    _assert_close(got, ref)


@pytest.mark.kernels_ci
def test_mask_digest_changes_with_mask(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=8, dim=32)
    mask_a = torch.ones(8, 8, dtype=torch.bool, device=cpu_device)
    mask_b = mask_a.clone()
    mask_b[:, -1] = False
    chain = kernel_mod.UnifiedReceiptChain()
    kernel_mod.receipt_attn(q, k, v, attn_mask=mask_a, chain=chain)
    kernel_mod.receipt_attn(q, k, v, attn_mask=mask_b, chain=chain)
    d0 = chain.receipts()[0]["body"]["mask_digest"]
    d1 = chain.receipts()[1]["body"]["mask_digest"]
    assert d0 != "none"
    assert d1 != "none"
    assert d0 != d1
    ok, depth, first_break = chain.verify()
    assert ok is True
    assert depth == 2
    assert first_break is None


@pytest.mark.kernels_ci
def test_head_dim_64_cpu(kernel_mod, cpu_device):
    q, k, v = _qkv(cpu_device, torch.float32, seq=33, dim=64)
    got = kernel_mod.receipt_attn(q, k, v, causal=True)
    ref = kernel_mod.sdpa_equivalent(q, k, v, causal=True)
    _assert_close(got, ref)
