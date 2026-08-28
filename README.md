---
library_name: kernels
license: apache-2.0
tags:
  - kernel
  - attention
  - triton
  - flash-attention
  - provenance
  - receipts
homepage: https://a-11-oy.com
---

# szl-receipt-attn

Original SZL Triton tiled fused attention with a SHA3-256 receipt chain.
FlashAttention *silhouette*, not a rehost.

**Owner:** Stephen P. Lutar Jr. / SZL Holdings
**Homepage:** [https://a-11-oy.com](https://a-11-oy.com)
**Canonical source:** [github.com/szl-holdings/szl-receipt-attn](https://github.com/szl-holdings/szl-receipt-attn)
**Hub:** [SZLHOLDINGS/szl-receipt-attn](https://huggingface.co/SZLHOLDINGS/szl-receipt-attn)

GitHub bytes **are** the artifact. Do not publish a Hub tree whose files
differ from this repository.

## Honesty plate

- **Doctrine v11 LOCKED**
- **Λ = Conjecture 1** (advisory, uniqueness **OPEN**, never proven trust)
- Original SZL construction in the fused-attention category. Inspired by
  Dao et al. FA1 https://arxiv.org/abs/2205.14135, FA2
  https://arxiv.org/abs/2307.08691, FA3 https://arxiv.org/abs/2407.08608.
  **NOT** a rehost of Dao-AILab or kernels-community flash-attn packages.
- **No speedup claims. No fabricated benchmarks.**
- A failed check stays failed. `selfcheck()` never fabricates a pass.

## Paths (which is which)

| Path | Label | Where it runs |
| --- | --- | --- |
| `triton_cuda` | Triton JIT tiled fused-attention kernel | **CUDA only** |
| `torch_reference` | Pure-PyTorch tiled online softmax | **CPU tests** (and any case where Triton cannot run) |

Triton does not run on CPU. The CPU path is a labeled reference, not a
compiled CPU backend, and not a silent stand-in for a GPU kernel.

## Load

```python
from kernels import get_kernel

attn = get_kernel(
    "SZLHOLDINGS/szl-receipt-attn",
    revision="main",
    trust_remote_code=True,
)
```

## Public API

```python
import torch

q = torch.randn(2, 4, 64, 32)
k = torch.randn(2, 4, 64, 32)
v = torch.randn(2, 4, 64, 32)

chain = attn.UnifiedReceiptChain()
out = attn.receipt_attn(
    q, k, v,
    causal=False,
    attn_mask=None,
    chain=chain,   # optional
    scale=None,    # default 1/sqrt(head_dim)
)
ok, depth, first_break = chain.verify()
report = attn.selfcheck()  # inspect the dict; do not assume a pass
```

- Layout: `(batch, heads, seq, head_dim)`.
- Grouped-query attention: `q.shape[1]` must be a multiple of `k.shape[1]`.
- `attn_mask`: bool (`True` = keep) or additive float, broadcastable to
  `(B, H, S_q, S_k)`.
- `causal=True` requires equal query/key lengths.
- Receipt body (when `chain` is passed): shapes, dtype, causal flag,
  mask digest (SHA3-256 of mask bytes, or `"none"`), scale, and path
  label. The digest is an integrity fingerprint, not a signature.

## Correctness tolerances

Compared against the in-repo `sdpa_equivalent` (matmul + softmax) and,
when available, `torch.nn.functional.scaled_dot_product_attention`.

| dtype | atol | rtol |
| --- | --- | --- |
| float32 | 1e-4 | 1e-4 |
| float16 | 2e-2 | 1e-2 |
| bfloat16 | 2e-2 | 1.6e-2 |

CUDA tests skip cleanly when no GPU is present. A skip is not a pass.

Fully-masked rows: this kernel returns zeros (defined). One-shot softmax
of all `-inf` may yield NaN. Tests do not use fully-masked rows.

## selfcheck

From a kernel-builder / Hub load:

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-receipt-attn", revision="main", trust_remote_code=True)
print(k.selfcheck())
```

From this source tree (no Hub download):

```bash
PYTHONPATH=torch-ext python -c "from szl_receipt_attn import selfcheck; import json; print(json.dumps(selfcheck(), indent=2))"
```

`ok` means every *executed* check passed. Skipped CUDA/Triton checks are
recorded as `skipped`, never as `pass`. `fabricated` is always `False`.

## Layout (kernel-builder edition 5)

```text
build.toml          # edition = 5, [torch-noarch], no kernel.backend = "triton"
CARD.md             # Hub card template
LICENSE             # Apache-2.0, Copyright 2026 SZL Holdings
flake.nix
torch-ext/szl_receipt_attn/   # Triton JIT + torch reference + receipts
tests/
README.md
```

Triton source lives in `torch-ext/`. This is a `[torch-noarch]` JIT
kernel. Do not set `kernel.backend = "triton"` (obsolete).

## Tests (kernel-builder)

kernel-builder testshell sets `LOCAL_KERNELS` and tests load via
`get_kernel` (not a fabricated Hub pass):

```bash
kernel-builder testshell
python -m pytest tests -m kernels_ci --tb=short
```

Source tree, labeled CPU reference (no GPU required):

```bash
SZL_SOURCE_TREE_TESTS=1 PYTHONPATH=torch-ext python -m pytest tests --tb=short
```

CUDA/Triton cases skip cleanly when no GPU is present. A skip is not a
pass. There is no `benchmarks/` directory and no speedup claim.

## Hypothesis (non-binding)

Online softmax + SRAM tiles is the silhouette. The SZL cut is the
receipt chain and honesty labels, not a faster FA clone.

## License

Apache-2.0. Copyright 2026 SZL Holdings.
