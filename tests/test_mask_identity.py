# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Receipt mask identity (szl-mask-id/v1): topology, determinism, versioning.

Source-tree tests: they import torch-ext/szl_receipt_attn directly, like
tests/test_receipt_attn.py, because they pin the identity format of this
revision rather than whatever build the Hub serves.
"""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

import torch

_TORCH_EXT = Path(__file__).resolve().parents[1] / "torch-ext"
sys.path.insert(0, str(_TORCH_EXT))

from szl_receipt_attn import ReceiptChain, receipt_attn  # noqa: E402
from szl_receipt_attn import attn as attn_mod  # noqa: E402

V1 = "szl-mask-id/v1"


def _legacy_unversioned(attn_mask, causal):
    """The unversioned pre-v1 sum-plus-shape fingerprint, reproduced only to prove the defect."""
    bits = f"causal={int(causal)}"
    if attn_mask is None:
        return bits + ":none"
    h = float(torch.sum(attn_mask.to(torch.float32)).item())
    return f"{bits}:sum={h:.6g}:shape={tuple(attn_mask.shape)}"


def _equal_count_pair():
    mask_a = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    mask_b = torch.tensor(
        [
            [True, False, False, True],
            [False, True, True, False],
            [True, True, False, False],
            [False, False, True, True],
        ]
    )
    return mask_a, mask_b


def test_equal_count_different_topology_gets_different_identity():
    mask_a, mask_b = _equal_count_pair()
    assert mask_a.shape == mask_b.shape
    assert int(mask_a.sum()) == int(mask_b.sum())
    assert not torch.equal(mask_a, mask_b)
    # The unversioned fingerprint collides on exactly this pair (the RQ-10 finding).
    assert _legacy_unversioned(mask_a, False) == _legacy_unversioned(mask_b, False)

    assert attn_mod._mask_digest(mask_a, False) != attn_mod._mask_digest(mask_b, False)

    q = k = v = torch.randn(1, 1, 4, 8)
    chain_a, chain_b = ReceiptChain(), ReceiptChain()
    receipt_attn(q, k, v, attn_mask=mask_a, chain=chain_a, prefer="torch")
    receipt_attn(q, k, v, attn_mask=mask_b, chain=chain_b, prefer="torch")
    assert chain_a._rows[0]["mask"] != chain_b._rows[0]["mask"]
    assert chain_a._rows[0]["digest"] != chain_b._rows[0]["digest"]
    assert chain_a.verify() == (True, 1, -1)
    assert chain_b.verify() == (True, 1, -1)


def test_permuted_float_bias_with_equal_sum_gets_different_identity():
    bias = torch.tensor([[0.0, -1.0, -2.0, -3.0]] * 4)
    permuted = bias.flip(-1)
    assert float(bias.sum()) == float(permuted.sum())
    assert _legacy_unversioned(bias, False) == _legacy_unversioned(permuted, False)
    assert attn_mod._mask_digest(bias, False) != attn_mod._mask_digest(permuted, False)


def test_identical_content_is_stable_and_layout_independent():
    mask_a, _ = _equal_count_pair()
    noncontiguous = mask_a.t().contiguous().t()
    assert torch.equal(mask_a, noncontiguous)
    assert not noncontiguous.is_contiguous()
    sliced = torch.zeros(6, 6, dtype=torch.bool)
    sliced[1:5, 2:6] = mask_a
    window = sliced[1:5, 2:6]
    assert window.storage_offset() != 0

    expected = attn_mod._mask_digest(mask_a, True)
    assert attn_mod._mask_digest(mask_a, True) == expected
    assert attn_mod._mask_digest(mask_a.clone(), True) == expected
    assert attn_mod._mask_digest(noncontiguous, True) == expected
    assert attn_mod._mask_digest(window, True) == expected


def test_v1_known_answer_bool_uses_uint8_zero_one_layout():
    mask = torch.tensor([[True, False], [False, True]])
    payload = bytes([1, 0, 0, 1])
    assert hashlib.sha256(payload).hexdigest() == (
        "afa7518106309c22d325df6d2663249d158d2f36f1976269d6d4104d9198a108"
    )
    assert attn_mod._mask_digest(mask, False) == (
        "szl-mask-id/v1;causal=0;dtype=bool;shape=2x2;"
        "sha256=afa7518106309c22d325df6d2663249d158d2f36f1976269d6d4104d9198a108"
    )


def test_v1_known_answer_float_uses_little_endian_element_bytes():
    bias = torch.tensor([[0.0, float("-inf")], [-0.5, 2.0]], dtype=torch.float32)
    payload = struct.pack("<4f", 0.0, float("-inf"), -0.5, 2.0)
    assert hashlib.sha256(payload).hexdigest() == (
        "2c92ef1a769022a95d06dab10d846da0d0323a56499de0ea8b9d803b885c5337"
    )
    assert attn_mod._mask_digest(bias, True) == (
        "szl-mask-id/v1;causal=1;dtype=float32;shape=2x2;"
        "sha256=2c92ef1a769022a95d06dab10d846da0d0323a56499de0ea8b9d803b885c5337"
    )


def test_dtype_shape_and_causal_are_bound():
    keep = torch.tensor([[True, False], [True, True]])
    as_float = keep.to(torch.float32)
    as_uint8 = keep.to(torch.uint8)
    ids = {
        attn_mod._mask_digest(keep, False),
        attn_mod._mask_digest(as_float, False),
        attn_mod._mask_digest(as_uint8, False),
        attn_mod._mask_digest(keep, True),
    }
    assert len(ids) == 4

    # Same canonical bytes, different logical shape.
    flat = torch.tensor([True, False, True, True, False, False, True, False])
    reshaped = {attn_mod._mask_digest(flat.reshape(s), False) for s in [(8,), (2, 4), (4, 2), (1, 2, 4)]}
    assert len(reshaped) == 4

    # Signed zero is different bytes, so a different identity (exact-byte binding).
    zero = torch.zeros(2, 2)
    assert attn_mod._mask_digest(zero, False) != attn_mod._mask_digest(-zero, False)


def test_no_mask_identity_is_versioned():
    assert attn_mod._mask_digest(None, False) == "szl-mask-id/v1;causal=0;mask=none"
    assert attn_mod._mask_digest(None, True) == "szl-mask-id/v1;causal=1;mask=none"


def test_legacy_and_v1_identities_never_compare_equal():
    assert attn_mod.MASK_IDENTITY_VERSION == V1
    mask_a, mask_b = _equal_count_pair()
    cases = [None, mask_a, mask_b, mask_a.to(torch.float32), torch.ones(3, 5, dtype=torch.bool)]
    for causal in (False, True):
        for mask in cases:
            new = attn_mod._mask_digest(mask, causal)
            old = _legacy_unversioned(mask, causal)
            assert new.startswith(V1 + ";")
            assert not old.startswith(V1)
            assert new != old

    q = k = v = torch.randn(1, 1, 4, 8)
    chain = ReceiptChain()
    receipt_attn(q, k, v, attn_mask=mask_a, chain=chain, prefer="torch")
    receipt_attn(q, k, v, causal=True, chain=chain, prefer="torch")
    assert [row["mask"].split(";", 1)[0] for row in chain._rows] == [V1, V1]


def test_legacy_receipt_chain_still_verifies_and_is_tagged_unversioned():
    """Old receipts stay chain-valid; their mask value is visibly pre-v1.

    The chain hashes the "mask" value opaquely, so a chain written with the
    unversioned fingerprint keeps verifying after this change. Nothing here
    upgrades or reinterprets it: a v1-aware reader sees no version tag and
    must treat it as an unsupported (legacy) identity.
    """
    mask_a, _ = _equal_count_pair()
    legacy = ReceiptChain()
    for mask, causal in ((mask_a, False), (None, True)):
        legacy.emit(
            {
                "op": "receipt_attn",
                "path": "torch_reference",
                "q_shape": [1, 1, 4, 8],
                "dtype": "float32",
                "causal": causal,
                "mask": _legacy_unversioned(mask, causal),
                "lambda": "Conjecture 1",
            }
        )
    assert legacy.verify() == (True, 2, -1)
    assert [row["mask"] for row in legacy._rows] == [
        "causal=0:sum=8:shape=(4, 4)",
        "causal=1:none",
    ]
    for row in legacy._rows:
        assert row["mask"].split(";", 1)[0] != V1

    # Rewriting a legacy value in place (e.g. "upgrading" it to v1) breaks the chain.
    legacy._rows[0]["mask"] = attn_mod._mask_digest(mask_a, False)
    assert legacy.verify() == (False, 0, 0)


def test_chunked_hash_equals_one_shot_hash(monkeypatch):
    gen = torch.Generator().manual_seed(1013)
    mask = torch.rand(3, 7, 5, generator=gen) > 0.5
    bias = torch.randn(3, 7, 5, generator=gen)
    one_shot_mask = attn_mod._mask_digest(mask, False)
    one_shot_bias = attn_mod._mask_digest(bias, False)
    monkeypatch.setattr(attn_mod, "_HASH_CHUNK_ELEMS", 4)
    assert attn_mod._mask_digest(mask, False) == one_shot_mask
    assert attn_mod._mask_digest(bias, False) == one_shot_bias
    expected = hashlib.sha256(struct.pack(f"<{bias.numel()}f", *bias.reshape(-1).tolist())).hexdigest()
    assert one_shot_bias.endswith(";sha256=" + expected)


def test_big_endian_host_path_swaps_each_element(monkeypatch):
    bias = torch.tensor([[0.0, float("-inf")], [-0.5, 2.0]], dtype=torch.float32)
    big_endian_bytes = struct.pack(">4f", 0.0, float("-inf"), -0.5, 2.0)
    monkeypatch.setattr(attn_mod, "_HASH_CHUNK_ELEMS", 3)
    if sys.byteorder == "little":
        # Pretending to be big-endian on a little-endian host must byte-swap each
        # 4-byte element, which yields big-endian bytes here; on a real
        # big-endian host the same swap yields the canonical little-endian bytes.
        monkeypatch.setattr(attn_mod.sys, "byteorder", "big")
        got = attn_mod._mask_digest(bias, False)
        assert got.endswith(";sha256=" + hashlib.sha256(big_endian_bytes).hexdigest())
    # bool/uint8 are one byte wide: no swap either way.
    mask = torch.tensor([[True, False], [False, True]])
    assert attn_mod._mask_digest(mask, False).endswith(
        ";sha256=afa7518106309c22d325df6d2663249d158d2f36f1976269d6d4104d9198a108"
    )


_CHILD = r"""
import sys
import torch
sys.path.insert(0, sys.argv[1])
from szl_receipt_attn import attn
gen = torch.Generator().manual_seed(20261013)
mask = torch.rand(2, 1, 16, 16, generator=gen) > 0.5
bias = torch.randn(1, 2, 8, 8, generator=gen)
print(attn._mask_digest(mask, True))
print(attn._mask_digest(bias, False))
"""


def test_identity_is_deterministic_across_processes():
    gen = torch.Generator().manual_seed(20261013)
    mask = torch.rand(2, 1, 16, 16, generator=gen) > 0.5
    bias = torch.randn(1, 2, 8, 8, generator=gen)
    expected = [attn_mod._mask_digest(mask, True), attn_mod._mask_digest(bias, False)]
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", _CHILD, str(_TORCH_EXT)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert out.stdout.split() == expected
