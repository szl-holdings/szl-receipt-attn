# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Load-path honesty. LOCAL_KERNELS must not skip a failed get_kernel."""

from __future__ import annotations

import os

import pytest


@pytest.mark.kernels_ci
def test_local_kernels_get_kernel_is_required():
    if not os.environ.get("LOCAL_KERNELS"):
        pytest.skip(
            "LOCAL_KERNELS unset — source-tree or Hub skip path is labeled elsewhere"
        )
    kernels = pytest.importorskip("kernels")
    kernels.get_kernel(
        "SZLHOLDINGS/szl-receipt-attn",
        revision="main",
        trust_remote_code=True,
    )
