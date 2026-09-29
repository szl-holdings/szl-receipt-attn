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

# szl-receipt-attn

Original Triton tiled fused attention (FlashAttention silhouette, not a copy). Each call can emit a SHA3-256 receipt onto an optional chain. Original SZL construction in the fused-attention category, inspired by Dao et al. FA1/FA2/FA3. **NOT** a rehost of Dao-AILab flash-attention, kernels-community/flash-attn2/3/4, hopper/, cute/, or any `.cu`. No speedup claim.

## Kernel

| Field | Value |
| --- | --- |
| Hub id | `SZLHOLDINGS/szl-receipt-attn` |
| Library | `kernels` |
| Backends | cpu (torch reference), cuda (original Triton tiles, untimed) |

```python
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", revision="main", trust_remote_code=True)
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

## Claims

| Label | Claim | Evidence |
| --- | --- | --- |
| MEASURED | CPU correctness. `tests/test_receipt_attn.py` asserts that causal fp32 output on the torch-reference path matches `torch.nn.functional.scaled_dot_product_attention` within atol=1e-5, rtol=1e-5, that a one-call receipt chain verifies, and that `selfcheck()` reports ok. CI runs it on every pull request and every push to main (`cpu-tests.yml`). | [receipt](https://github.com/szl-holdings/szl-receipt-attn/blob/6bd6a7eb1c6ae9703b87afed0a237a9f5a03eef3/tests/test_receipt_attn.py) |
| MEASURED | API edges. `tests/test_api_edges.py` asserts the same fp32 band against SDPA for a custom scale, unequal query and key lengths, and 4-D boolean masks; that the receipt's mask digest changes with the mask; that head dim 64 runs `torch_reference`; and that `prefer="triton"` on CPU records `torch_reference_fallback`. | [receipt](https://github.com/szl-holdings/szl-receipt-attn/blob/6bd6a7eb1c6ae9703b87afed0a237a9f5a03eef3/tests/test_api_edges.py) |
| REPORTED | CPU `get_kernel` load from the Kernel Hub (kernels 0.16.1, `build/torch-universal` and `build/torch-cpu`, `selfcheck` ok) and a local pytest run, recorded 2026-08-29 in the Hub-only `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` against source commit 1aa6cf77. No receipt for them is committed in this repository. | none linked |
| UNAVAILABLE | CUDA / Triton timing. The Triton path exists in source but has no timed GPU run and no cubin. CI has no GPU, so the CUDA path test is skipped there. | none linked |
| NOT_CLAIMED | Any speed-up, tokens/s or energy figure. | none linked |

Labels follow the [SZL claim language](https://github.com/szl-holdings/.github/blob/main/docs/CLAIM_LANGUAGE.md). A claim is only as strong as the receipt it links.

## Limits

- Kernel, not weights.
- The Triton path covers CUDA inputs without `attn_mask` and head dim ≤ 32 only; everything else runs the torch reference.
- Receipts prove integrity and declared origin only; they do not prove accuracy, readiness or performance.

## Source and provenance

| Field | Value |
| --- | --- |
| Source repository | [szl-holdings/szl-receipt-attn](https://github.com/szl-holdings/szl-receipt-attn) |
| Proof | <https://github.com/szl-holdings/szl-receipt-attn> |
| License | `apache-2.0` |

This card is written to the Hub only by the committed mirror workflow of szl-holdings/szl-receipt-attn.
