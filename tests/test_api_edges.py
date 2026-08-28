# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Extra correctness tests against KERNEL original API. No benchmarks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

# KERNEL documented fp32 band vs torch SDPA.
_ATOL = 1.0e-5
_RTOL = 1.0e-5


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=_ATOL, rtol=_RTOL
    )


def _sdpa(q, k, v, *, causal=False, attn_mask=None, scale=None):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=scale
    )


@pytest.mark.kernels_ci
def test_custom_scale(kernel_mod, cpu_device):
    torch.manual_seed(2026)
    q = torch.randn(2, 4, 16, 32, device=cpu_device)
    k = torch.randn(2, 4, 16, 32, device=cpu_device)
    v = torch.randn(2, 4, 16, 32, device=cpu_device)
    scale = 0.25
    got = kernel_mod.receipt_attn(q, k, v, scale=scale, prefer="torch")
    _assert_close(got, _sdpa(q, k, v, scale=scale))


@pytest.mark.kernels_ci
def test_unequal_seq_non_causal(kernel_mod, cpu_device):
    torch.manual_seed(11)
    q = torch.randn(1, 2, 12, 32, device=cpu_device)
    k = torch.randn(1, 2, 20, 32, device=cpu_device)
    v = torch.randn(1, 2, 20, 32, device=cpu_device)
    got = kernel_mod.receipt_attn(q, k, v, causal=False, prefer="torch")
    _assert_close(got, _sdpa(q, k, v, causal=False))


@pytest.mark.kernels_ci
def test_four_d_bool_mask(kernel_mod, cpu_device):
    torch.manual_seed(7)
    q = torch.randn(2, 4, 16, 32, device=cpu_device)
    k = torch.randn(2, 4, 16, 32, device=cpu_device)
    v = torch.randn(2, 4, 16, 32, device=cpu_device)
    mask = torch.ones(2, 4, 16, 16, dtype=torch.bool, device=cpu_device)
    mask[:, :, :, 12:] = False
    got = kernel_mod.receipt_attn(q, k, v, attn_mask=mask, prefer="torch")
    _assert_close(got, _sdpa(q, k, v, attn_mask=mask))


@pytest.mark.kernels_ci
def test_mask_field_changes_with_mask(kernel_mod, cpu_device):
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8, 32, device=cpu_device)
    k = torch.randn(1, 2, 8, 32, device=cpu_device)
    v = torch.randn(1, 2, 8, 32, device=cpu_device)
    mask_a = torch.ones(8, 8, dtype=torch.bool, device=cpu_device)
    mask_b = mask_a.clone()
    mask_b[:, -1] = False
    chain = kernel_mod.ReceiptChain()
    kernel_mod.receipt_attn(q, k, v, attn_mask=mask_a, chain=chain, prefer="torch")
    kernel_mod.receipt_attn(q, k, v, attn_mask=mask_b, chain=chain, prefer="torch")
    d0 = chain._rows[0]["mask"]
    d1 = chain._rows[1]["mask"]
    assert d0 != d1
    ok, depth, brk = chain.verify()
    assert ok is True
    assert depth == 2
    assert brk == -1


@pytest.mark.kernels_ci
def test_head_dim_64_uses_torch_reference(kernel_mod, cpu_device):
    torch.manual_seed(64)
    q = torch.randn(1, 2, 16, 64, device=cpu_device)
    k = torch.randn(1, 2, 16, 64, device=cpu_device)
    v = torch.randn(1, 2, 16, 64, device=cpu_device)
    chain = kernel_mod.ReceiptChain()
    got = kernel_mod.receipt_attn(q, k, v, causal=True, chain=chain, prefer="auto")
    _assert_close(got, _sdpa(q, k, v, causal=True))
    assert chain._rows[0]["path"] == "torch_reference"


@pytest.mark.kernels_ci
def test_prefer_triton_on_cpu_falls_back(kernel_mod, cpu_device):
    torch.manual_seed(9)
    q = torch.randn(1, 2, 16, 32, device=cpu_device)
    k = torch.randn(1, 2, 16, 32, device=cpu_device)
    v = torch.randn(1, 2, 16, 32, device=cpu_device)
    chain = kernel_mod.ReceiptChain()
    got = kernel_mod.receipt_attn(q, k, v, chain=chain, prefer="triton")
    _assert_close(got, _sdpa(q, k, v))
    assert chain._rows[0]["path"] == "torch_reference_fallback"


@pytest.mark.kernels_ci
def test_cuda_path_when_gpu_present(kernel_mod, cuda_device):
    """Triton path only on CUDA. Honest skip if no GPU — not a pass."""
    torch.manual_seed(1)
    q = torch.randn(1, 2, 16, 32, device=cuda_device)
    k = torch.randn(1, 2, 16, 32, device=cuda_device)
    v = torch.randn(1, 2, 16, 32, device=cuda_device)
    chain = kernel_mod.ReceiptChain()
    y = kernel_mod.receipt_attn(q, k, v, causal=True, chain=chain, prefer="auto")
    assert y.shape == q.shape
    assert chain._rows[0]["path"] in ("triton", "torch_reference_fallback")
