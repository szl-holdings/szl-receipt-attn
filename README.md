# szl-receipt-attn
<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

Canonical GitHub source for `SZLHOLDINGS/szl-receipt-attn`.

Original Triton tiled fused attention (FlashAttention silhouette, not a copy).
Each call can emit a SHA3-256 receipt onto an optional chain.

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN).  
Original SZL construction in the fused-attention category. Inspired by Dao et al. FA1/FA2/FA3. **NOT** a rehost of Dao-AILab flash-attention, kernels-community/flash-attn2/3/4, hopper/, cute/, or any `.cu`.

GitHub bytes are the artifact. The Hub is the publish target, written only by this
repository's [`hf-kernel-publish`](.github/workflows/hf-kernel-publish.yml) workflow.

## Load

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", version=1, trust_remote_code=True)
```

`version=1` resolves the `v1` branch of the Kernel Hub repo
[`kernels/SZLHOLDINGS/szl-receipt-attn`](https://huggingface.co/kernels/SZLHOLDINGS/szl-receipt-attn).
`hf-kernel-publish` builds it from `torch-ext/szl_receipt_attn` at one commit of this
repository, publishes it, and then loads it through `get_kernel` in a fresh environment;
the run fails if the served bytes differ from GitHub. `SZL_KERNEL_SOURCE.json` on the Hub
names that commit. The model-type repo `SZLHOLDINGS/szl-receipt-attn` is only a pointer.

`trust_remote_code=True` permits execution of the kernel repository's Python; the
narrowest form is `trust_remote_code=["SZLHOLDINGS/szl-receipt-attn"]`. For production,
pin the `v1` branch's current kernel-repo commit with `revision=<commit>`: the
[kernel card](CARD.md) shows how. GitHub source commits and model-repo commits are not
kernel revisions. Publication details are in [hf/README.md](hf/README.md).

## Source-only development

Review [`torch-ext/szl_receipt_attn/`](https://github.com/szl-holdings/szl-receipt-attn/tree/010519f0c26906be3ecf07bd91c969e2295eb5d4/torch-ext/szl_receipt_attn)
at the GitHub commit you intend to run, separately from any Hub release.
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
