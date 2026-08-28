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


def test_selfcheck():
    r = selfcheck()
    assert r["ok"] is True
    assert r["lambda"] == "Conjecture 1"
