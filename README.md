# szl-receipt-attn

Canonical GitHub source for `SZLHOLDINGS/szl-receipt-attn`.

Original Triton tiled fused attention (FlashAttention silhouette, not a copy).
Each call can emit a SHA3-256 receipt onto an optional chain.

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN).  
Original SZL construction in the fused-attention category. Inspired by Dao et al. FA1/FA2/FA3. **NOT** a rehost of Dao-AILab flash-attention, kernels-community/flash-attn2/3/4, hopper/, cute/, or any `.cu`.

GitHub bytes are the artifact. Hub is the publish mirror. ATELIER owns Hub cards.

## Load

```python
from kernels import get_kernel

attn = get_kernel(
    "SZLHOLDINGS/szl-receipt-attn",
    revision="main",
    trust_remote_code=True,
)
```

```python
import torch
from szl_receipt_attn import receipt_attn, ReceiptChain, selfcheck

q = k = v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
y = receipt_attn(q, k, v, causal=True, chain=chain)
print(chain.verify(), selfcheck())
```

`selfcheck()` never fabricates a pass. It runs a small CPU torch-reference check.

## Paths (honesty)

| `path` | When |
|---|---|
| `triton` | CUDA, no `attn_mask`, head dim ≤ 32, original SZL Triton tiles |
| `torch_reference` | CPU, or `prefer="torch"`, or mask / head dim > 32 |
| `torch_reference_fallback` | Triton was selected and raised |

No speedup claims. No fabricated CUDA benches in this repo.

## Correctness band (documented, not a bench)

fp32 vs `torch.nn.functional.scaled_dot_product_attention`: **atol=1e-5, rtol=1e-5**.

## Tests

- kernel-builder: `nix run .#testshell-torch-ext-local` (sets `LOCAL_KERNELS`; `get_kernel` must hard-fail if that env is ignored)
- source tree (labeled, not a Hub load): `SZL_SOURCE_TREE_TESTS=1 PYTHONPATH=torch-ext python -m pytest tests/ -q`

## License

Apache-2.0. Copyright 2026 SZL Holdings. Owner: Stephen P. Lutar Jr. / SZL Holdings. Homepage: https://a-11-oy.com
