---
license: apache-2.0
library_name: kernels
tags:
- kernels
- kernel
- fused-attention
- receipt
- szl-holdings
szl:
  source_repo: szl-holdings/szl-receipt-attn
  proof_url: https://github.com/szl-holdings/szl-receipt-attn
---
<!-- hf-card: type=kernel source=szl-holdings/szl-receipt-attn vars=hf/card.yaml -->
<!-- Rendered by szl-holdings/.github hf-card/render.py. Edit hf/card.yaml in szl-holdings/szl-receipt-attn; do not edit this card on the Hub. -->

<!-- Native artwork: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon SHA-256 6e2b7d72a3af3d9602b79de41df0194ede6496a9661f057108ec224c0efc92e6. -->
<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="40" height="40" /></a></p>

<p><small>Orbit v2 artwork © SZL Holdings · <a href="https://github.com/szl-holdings/szl-brand/blob/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/LICENSE">CC BY 4.0</a>. Artifact and documentation licenses remain as declared by their source.</small></p>

# szl-receipt-attn

Original Triton tiled fused attention (FlashAttention silhouette, not a copy). Each call can emit a SHA3-256 receipt onto an optional chain. Original SZL construction in the fused-attention category, inspired by Dao et al. FA1/FA2/FA3. **NOT** a rehost of Dao-AILab flash-attention, kernels-community/flash-attn2/3/4, hopper/, cute/, or any `.cu`. No speedup claim.

**Artifact:** Kernel · **Stage:** Not declared

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-receipt-attn) · [Evidence](https://github.com/szl-holdings/szl-receipt-attn)

## Before you use it

- Kernel, not weights.
- The Triton path covers CUDA inputs without `attn_mask` and head dim ≤ 32 only; everything else runs the torch reference.
- Receipts prove integrity and declared origin only; they do not prove accuracy, readiness or performance.

<details>
<summary>Technical details and evidence</summary>

## Kernel

| Field | Value |
| --- | --- |
| Hub id | `SZLHOLDINGS/szl-receipt-attn` |
| Library | `kernels` |
| Backends | cpu (torch reference), cuda (original Triton tiles, untimed) |

```python
# pip install "kernels>=0.17.2" torch
# This loads and executes remote Python code; review the published source first.
from kernels import get_kernel

# version=1 resolves the v1 branch of the kernel repo (the stable API channel).
attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", version=1, trust_remote_code=True)
```

## Doctrine

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN, never a theorem). GitHub bytes are the artifact; the Hub is the publish mirror.

## Paths (honesty)

| `path` | When |
|---|---|
| `triton` | CUDA, no `attn_mask`, head dim ≤ 32, original SZL Triton tiles |
| `torch_reference` | CPU, or `prefer="torch"`, or mask / head dim > 32 |
| `torch_reference_fallback` | Triton was selected and raised |

The returned `path` says which one ran. No fabricated CUDA benches.

## Local source tree

Put `torch-ext/` on `PYTHONPATH`, then:

```python
import torch
from szl_receipt_attn import receipt_attn, ReceiptChain, selfcheck

q = k = v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
y = receipt_attn(q, k, v, causal=True, chain=chain)
print(chain.verify(), selfcheck())
```

`selfcheck()` never fabricates a pass. It runs a small CPU torch-reference check.

## Loading and pinning

- `version=1` resolves the `v1` branch of this kernel repository. It is republished
  from GitHub on every release; a breaking API change starts `v2` and leaves `v1` frozen.
  The `kernels` library refuses a call that names neither a version nor a revision.
- `trust_remote_code` is required: SZLHOLDINGS is not a trusted kernel publisher on the
  Hub. The narrowest form is `trust_remote_code=["SZLHOLDINGS/szl-receipt-attn"]`.
- CPU hosts load `build/torch-cpu`; CUDA and other hosts load `build/torch-universal`.
  Both variants carry identical code, built from the same GitHub commit.

To pin an exact build for production, resolve the `v1` branch to a commit of this
kernel repository and record it in your lockfile:

```python
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("SZLHOLDINGS/szl-receipt-attn", repo_type="kernel")
pin = next(b.target_commit for b in refs.branches if b.name == "v1")
attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", revision=pin,
                  trust_remote_code=["SZLHOLDINGS/szl-receipt-attn"])
```

A GitHub source commit or a commit of the model-type twin is not a kernel revision and
does not resolve here.

## Source and Hub release scope

This repository is written only by the `hf-kernel-publish` workflow of
[szl-holdings/szl-receipt-attn](https://github.com/szl-holdings/szl-receipt-attn).
It builds `build/torch-cpu` and `build/torch-universal` from `torch-ext/szl_receipt_attn`
at one GitHub commit, publishes them with this card and `LICENSE` to `main` and `v1`,
and then loads the result through `get_kernel` in a fresh environment; the run fails
if the served bytes differ from GitHub. `SZL_KERNEL_SOURCE.json` here names the
GitHub commit the build came from. Publication receipts are artifacts of that
workflow's runs; they are not committed to this card.

The Hub-only plates `OPERATIONAL.json` and `BENCH.laptop-blackwell.json` (2026-08-29,
kernels 0.16.1, source commit 1aa6cf77) described a superseded build and were removed
by the first canonical publish. They remain in this kernel repository's Git history.

Publication context is recorded in [hf/README.md](https://github.com/szl-holdings/szl-receipt-attn/blob/main/hf/README.md).

## Evidence record

**MEASURED** — CPU correctness. `tests/test_receipt_attn.py` asserts that causal fp32 output on the torch-reference path matches `torch.nn.functional.scaled_dot_product_attention` within atol=1e-5, rtol=1e-5, that a one-call receipt chain verifies, and that `selfcheck()` reports ok. CI runs it on every pull request and every push to main (`cpu-tests.yml`). [Receipt](https://github.com/szl-holdings/szl-receipt-attn/blob/6bd6a7eb1c6ae9703b87afed0a237a9f5a03eef3/tests/test_receipt_attn.py).

**MEASURED** — API edges. `tests/test_api_edges.py` asserts the same fp32 band against SDPA for a custom scale, unequal query and key lengths, and 4-D boolean masks; that the receipt's mask digest changes with the mask; that head dim 64 runs `torch_reference`; and that `prefer="triton"` on CPU records `torch_reference_fallback`. [Receipt](https://github.com/szl-holdings/szl-receipt-attn/blob/6bd6a7eb1c6ae9703b87afed0a237a9f5a03eef3/tests/test_api_edges.py).

**UNAVAILABLE** — CUDA / Triton timing. The Triton path exists in source but has no timed GPU run and no cubin. CI has no GPU, so the CUDA path test is skipped there. (no receipt linked).

**NOT_CLAIMED** — Any speed-up, tokens/s or energy figure. (no receipt linked).

Labels follow the [SZL claim language](https://github.com/szl-holdings/.github/blob/main/docs/CLAIM_LANGUAGE.md). A claim is only as strong as the receipt it links.

## Source and provenance

| Field | Value |
| --- | --- |
| Source repository | [szl-holdings/szl-receipt-attn](https://github.com/szl-holdings/szl-receipt-attn) |
| Proof | <https://github.com/szl-holdings/szl-receipt-attn> |
| License | `apache-2.0` |

Rendering this card does not establish Hub publication. Verify any Hub copy against its source revision, publication evidence, and exact Hub revision.

</details>
