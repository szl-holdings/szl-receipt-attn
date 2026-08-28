---
language:
- Python
license: apache-2.0
tags:
- kernels
- fused-attention
- receipt
library_name: kernels
---

# {{ gpu }}

Original SZL construction in the fused-attention category.
Inspired by Dao et al. FA1/FA2/FA3. **Not** a rehost of Dao-AILab or kernels-community/flash-attn2/3/4.

GitHub is source of truth. Hub is the publish mirror. ATELIER owns Hub cards.
Doctrine v11. Λ = Conjecture 1 (never a theorem). No speedup claim.
CPU: torch reference (`path="torch_reference"`). CUDA: original Triton tiles + online softmax (`path="triton"`, head dim <= 32 in v0). Triton exceptions fall back to torch (`path="torch_reference_fallback"`).
Sage INT8/FP8 is a fourth kernel, ROADMAP, not this package.

Apache-2.0. Copyright 2026 SZL Holdings.

## Load

```python
from kernels import get_kernel
attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", revision="main", trust_remote_code=True)
```
