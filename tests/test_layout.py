# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Kernel-builder layout checks. These are not performance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.kernels_ci
def test_build_toml_edition_5_torch_noarch():
    text = (_ROOT / "build.toml").read_text(encoding="utf-8")
    assert "edition = 5" in text
    assert "[torch-noarch]" in text
    assert "[kernel." not in text
    assigned_triton = [
        line
        for line in text.splitlines()
        if "triton" in line.lower() and not line.lstrip().startswith("#")
    ]
    assert assigned_triton == []
    assert "repo-id = \"SZLHOLDINGS/szl-receipt-attn\"" in text


@pytest.mark.kernels_ci
def test_no_benchmarks_directory():
    """Honesty: this package does not ship fabricated benches."""
    assert not (_ROOT / "benchmarks").exists()


@pytest.mark.kernels_ci
def test_required_kernel_builder_files():
    for rel in (
        "build.toml",
        "CARD.md",
        "LICENSE",
        "README.md",
        "flake.nix",
        "torch-ext/szl_receipt_attn/__init__.py",
        "torch-ext/szl_receipt_attn/triton_kernel.py",
        "tests/test_receipt_attn.py",
    ):
        assert (_ROOT / rel).is_file(), rel
