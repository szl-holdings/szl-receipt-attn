---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}tags:
  - kernel
  - attention
  - triton
  - flash-attention
  - provenance
  - receipts
homepage: https://a-11-oy.com
---

# {{ repo_id | default("SZLHOLDINGS/szl-receipt-attn") }}

Original SZL Triton tiled fused attention with a SHA3-256 receipt chain.
FlashAttention *silhouette*, not a rehost.

**Owner:** Stephen P. Lutar Jr. / SZL Holdings
**Homepage:** https://a-11-oy.com
**Canonical source:** https://github.com/szl-holdings/szl-receipt-attn

GitHub bytes **are** the artifact. Do not publish a Hub tree whose files
differ from the GitHub source.

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
compiled CPU backend.

## How to use

```python
# pip install -U kernels torch
from kernels import get_kernel

# If the org / user isn't a trusted publisher, pass trust_remote_code=True.
kernel_module = get_kernel("SZLHOLDINGS/szl-receipt-attn", revision="main", trust_remote_code=True)
out = kernel_module.receipt_attn(q, k, v, causal=False, attn_mask=None, chain=None, scale=None)
report = kernel_module.selfcheck()  # inspect; do not assume a pass
```

{% if functions %}
## Available functions
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% endif %}

## Correctness tolerances

| dtype | atol | rtol |
| --- | --- | --- |
| float32 | 1e-4 | 1e-4 |
| float16 | 2e-2 | 1e-2 |
| bfloat16 | 2e-2 | 1.6e-2 |

CUDA tests skip cleanly when no GPU is present. A skip is not a pass.

## Benchmarks

{% if has_benchmark %}
Benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
This card still makes **no speedup claim**.
{% else %}
No benchmark available. None is claimed.
{% endif %}
