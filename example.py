# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Example load of szl-receipt-attn. Not a benchmark. Shapes only."""

from __future__ import annotations

import torch
from kernels import get_kernel

attn = get_kernel(
    "SZLHOLDINGS/szl-receipt-attn",
    revision="main",
    trust_remote_code=True,
)

q = torch.randn(1, 2, 16, 32)
k = torch.randn(1, 2, 16, 32)
v = torch.randn(1, 2, 16, 32)
chain = attn.UnifiedReceiptChain()
out = attn.receipt_attn(q, k, v, causal=True, chain=chain)
ok, depth, first_break = chain.verify()
print("out", tuple(out.shape), "verify", ok, depth, first_break)
print("selfcheck fabricated", attn.selfcheck()["fabricated"])
