# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Portability-reference conformance tests (stdlib only, CPU honest).

These tests do not import torch, do not load the Hub kernel and make
no GPU claims. They check that the tile-level portability reference
mirrors the eager reference within the documented tolerance band,
that the causal mask is honored by every backend, and that receipt
hashing fails closed on mutation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PORT = Path(__file__).resolve().parents[1] / "portability" / "tk_tile_reference.py"


def _load():
    spec = importlib.util.spec_from_file_location("tk_tile_reference", _PORT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_tiled_matches_eager_within_band():
    mod = _load()
    q, k, v = mod.make_inputs()
    for causal in (True, False):
        out_t = mod.attention(q, k, v, causal, "python_tiled_flash")
        out_e = mod.attention(q, k, v, causal, "python_eager")
        assert mod.max_abs_diff(out_t, out_e) <= mod.ATOL


def test_causal_mask_honored_by_both_backends():
    mod = _load()
    q, k, v = mod.make_inputs()
    for backend in ("python_tiled_flash", "python_eager"):
        causal = mod.attention(q, k, v, True, backend)
        full = mod.attention(q, k, v, False, backend)
        assert mod.max_abs_diff(causal, full) > 0.0


def test_receipts_hash_chain_and_fail_closed():
    mod = _load()
    result = mod.run_conformance()
    assert result["verdict"] == "PASS"
    first, second = result["receipts"]
    assert first["sha256"] == mod.receipt_id(first)
    assert second["chain_prev_sha256"] == first["sha256"]
    mutated = dict(first)
    mutated["verdict"] = "FAIL"
    assert mod.receipt_id(mutated) != first["sha256"]
