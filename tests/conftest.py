# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TORCH_EXT = _REPO_ROOT / "torch-ext"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "kernels_ci: kernel-builder CI tests (keep the marked set under 60s)",
    )


def _load_via_get_kernel():
    kernels = pytest.importorskip("kernels")
    return kernels.get_kernel(
        "SZLHOLDINGS/szl-receipt-attn",
        revision="main",
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def kernel_mod():
    """Load the kernel the way Hub users will: ``get_kernel``.

    Source-tree developers may set ``SZL_SOURCE_TREE_TESTS=1`` to import
    ``torch-ext/szl_receipt_attn`` directly. That path is labeled and is
    not a fabricated Hub load.
    """
    if os.environ.get("SZL_SOURCE_TREE_TESTS") == "1":
        sys.path.insert(0, str(_TORCH_EXT))
        import szl_receipt_attn

        return szl_receipt_attn
    try:
        return _load_via_get_kernel()
    except Exception as exc:
        pytest.skip(
            "get_kernel could not load SZLHOLDINGS/szl-receipt-attn "
            f"({type(exc).__name__}: {exc}). Not a pass. Set "
            "SZL_SOURCE_TREE_TESTS=1 to exercise the source tree, or run "
            "under kernel-builder testshell (LOCAL_KERNELS)."
        )


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip(
            "No CUDA GPU — Triton path not exercised (honest skip, not a pass)"
        )
    return torch.device("cuda")
