# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

import torch
import torch.nn.functional as F

from szl_receipt_attn import ReceiptChain, receipt_attn, selfcheck


def test_matches_sdpa_cpu():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    y = receipt_attn(q, k, v, causal=True, prefer="torch")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def test_receipt_chain():
    q = k = v = torch.randn(1, 1, 4, 8)
    chain = ReceiptChain()
    receipt_attn(q, k, v, chain=chain, prefer="torch")
    ok, depth, brk = chain.verify()
    assert ok and depth == 1 and brk == -1


def test_receipt_snapshots_nested_caller_payload():
    payload = {"op": "receipt_attn", "q_shape": [1, 2, 8, 16],
               "metadata": {"labels": ["synthetic"], "options": {"scale": 0.25}}}
    chain = ReceiptChain()
    digest = chain.emit(payload)

    payload["q_shape"][0] = 99
    payload["metadata"]["labels"].append("later")
    payload["metadata"]["options"]["scale"] = 0.5

    assert chain._rows[0]["q_shape"] == [1, 2, 8, 16]
    assert chain._rows[0]["metadata"] == {
        "labels": ["synthetic"], "options": {"scale": 0.25}}
    assert chain.head() == digest
    assert chain.verify() == (True, 1, -1)
    assert "seq" not in payload and "prev" not in payload and "digest" not in payload


def test_reusing_payload_records_distinct_snapshots_without_breaking_chain():
    payload = {"q_shape": [1, 2, 8, 16]}
    chain = ReceiptChain()
    first = chain.emit(payload)
    payload["q_shape"][2] = 32
    second = chain.emit(payload)
    payload["q_shape"][2] = 64

    assert chain._rows[0]["q_shape"] == [1, 2, 8, 16]
    assert chain._rows[1]["q_shape"] == [1, 2, 32, 16]
    assert chain._rows[1]["prev"] == first
    assert chain.head() == second
    assert chain.verify() == (True, 2, -1)


def test_mutating_stored_nested_receipt_still_breaks_verification():
    payload = {"metadata": {"labels": ["synthetic"]}}
    chain = ReceiptChain()
    chain.emit(payload)
    chain._rows[0]["metadata"]["labels"].append("tampered")

    assert chain.verify() == (False, 0, 0)
    assert payload == {"metadata": {"labels": ["synthetic"]}}


def test_selfcheck():
    r = selfcheck()
    assert r["ok"] is True
    assert r["lambda"] == "Conjecture 1"
