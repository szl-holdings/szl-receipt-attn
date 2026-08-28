# szl-receipt-attn

Canonical GitHub source for `SZLHOLDINGS/szl-receipt-attn`.

Original Triton tiled fused attention (FlashAttention silhouette, not a copy).
Each call can emit a SHA3-256 receipt onto an optional chain.

```python
import torch
from szl_receipt_attn import receipt_attn, ReceiptChain, selfcheck

q = k = v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
y = receipt_attn(q, k, v, causal=True, chain=chain)
print(chain.verify(), selfcheck())
```

`get_kernel("SZLHOLDINGS/szl-receipt-attn", revision="main", trust_remote_code=True)` after publish.

Not Dao-AILab. No CUDA bench numbers. Λ = Conjecture 1. Apache-2.0.
