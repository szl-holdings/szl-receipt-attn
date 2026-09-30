# szl-receipt-attn
<!-- szl:header v1 -->
<!-- badges: add this repo's CI / release / status badges here -->
[![org: szl-holdings](https://img.shields.io/badge/org-szl--holdings-black)](https://github.com/szl-holdings)
[![doctrine](https://img.shields.io/badge/doctrine-control%20before%20action%20%C2%B7%20evidence%20after-blue)](https://a-11-oy.com)

**Control before action. Evidence after.**

Part of the [szl-holdings](https://github.com/szl-holdings) estate ·
Product: [a-11-oy.com](https://a-11-oy.com) ·
Proof: [a11oy.net](https://a11oy.net)
<!-- /szl:header -->

Canonical GitHub source for `SZLHOLDINGS/szl-receipt-attn`.

Original Triton tiled fused attention (FlashAttention silhouette, not a copy).
Each call can emit a SHA3-256 receipt onto an optional chain.

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN).  
Original SZL construction in the fused-attention category. Inspired by Dao et al. FA1/FA2/FA3. **NOT** a rehost of Dao-AILab flash-attention, kernels-community/flash-attn2/3/4, hopper/, cute/, or any `.cu`.

GitHub bytes are the artifact. Hub is the publish mirror. ATELIER owns Hub cards.

## Load

The [staged card and publisher plan](hf/README.md) records that the Hub build
predates this source tree. The current source APIs require a separately
verified matching build publication; a reachable Hub repo does not establish it.

Set `SZL_RECEIPT_ATTN_HF_REVISION` to the immutable **first-class Kernel Hub** commit
from a verified publication of [`kernels/SZLHOLDINGS/szl-receipt-attn`](https://huggingface.co/kernels/SZLHOLDINGS/szl-receipt-attn). Use the `kernels`
client version qualified with that publication. The GitHub source commit,
model-type mirror commit, and Kernel Hub commit are separate identities.
An observed head, a branch name, or a successful import does not qualify a release.

`trust_remote_code=True` permits execution of the selected repository's Python.
Review that exact revision, its provenance and publication evidence before enabling it.
The format check below only rejects missing or mutable revision inputs; it does not
verify hashes, publisher authorization or compatibility. If that evidence is unavailable,
stop the Hub load and use separately reviewed local source for development.

```python
import os
import re

hf_revision = os.environ.get("SZL_RECEIPT_ATTN_HF_REVISION", "")
if re.fullmatch(r"[0-9a-f]{40}", hf_revision) is None:
    raise ValueError("A verified immutable Kernel Hub revision is required")

from kernels import get_kernel

attn = get_kernel(
    "SZLHOLDINGS/szl-receipt-attn",
    revision=hf_revision,
    trust_remote_code=True,
)
```

## Source-only development

Review [`torch-ext/szl_receipt_attn/`](https://github.com/szl-holdings/szl-receipt-attn/tree/010519f0c26906be3ecf07bd91c969e2295eb5d4/torch-ext/szl_receipt_attn)
at that immutable GitHub source revision, separately from any Hub release.
With the source's dependencies already available, run from the reviewed checkout root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("torch-ext").resolve()))
import szl_receipt_attn as local_kernel
```

This selects local Python source rather than calling the Hub loader. Importing local
source also executes Python. This documentation check does not run that import,
install dependencies, qualify a runtime or establish a Hub publication.

Local API example:

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
