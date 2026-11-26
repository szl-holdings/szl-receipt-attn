# Portability tile reference (Cluster C)

Absorb record for the OUROBOROS round-2 leader pack, Cluster C
(kernels/ML/attention, consumer-GPU branch): a ThunderKittens-style
tile abstraction as the *portability reference* for the
``szl-receipt-attn`` Triton tile schedule. Leader sources studied:
the ThunderKittens 2.0 tile abstraction (Hazy Research, 2026-02-19),
the Vera Rubin NVL72 port (Together AI), and the FlashAttention-4
asymmetric-hardware posture (backend specialization with a portable
reference fallback, fail-closed to the reference kernel, never
receipt-attesting speed as correctness).

## What this is

``portability/tk_tile_reference.py`` expresses the flash-style
online-softmax tile schedule at backend-neutral tile level: iterate Q
tiles, stream K/V tiles, keep per-row running max / normalizer /
accumulator, rescale every tile. The schedule mirrors
``torch-ext/szl_receipt_attn``; only tile math is expressed here,
no memory layout or backend intrinsics.

Two reference backends implement the same schedule:

| backend | role |
|---|---|
| ``python_tiled_flash`` | tile scheduled, online softmax (the portability reference) |
| ``python_eager`` | full-matrix reference acceptance oracle |

Acceptance criterion (leader-pack test): one kernel schedule, two
backends, identical numeric tolerance, one hash-chained receipt per
backend. CPU-honest: fp64 host CPU, pure stdlib. No GPU timing, no
Triton cubin, no speedup claims. Receipts are fail-closed: a
tolerance mismatch yields verdict FAIL and a nonzero exit code.

Documented band: tiled-vs-eager ``atol=1e-9`` (fp64). The measured
device run (2026-11-26) passed at ``max_abs_diff=1.18e-16`` with
receipts published on
``SZLHOLDINGS/wave-logs`` at
``mesh/2026-11-26/ouroboros/tk-portability-clusterC/``.

## What this is NOT

- Not a benchmark; wall-clock is not recorded.
- Not a GPU qualification; the Triton path remains UNAVAILABLE for
  timing per the kernel card status table.
- Not a Hub publication or a Hub-load qualification; GitHub bytes
  remain the artifact.
- Not a PyPI release-path change.

Raising a new hardware backend against this reference means: match
the tile schedule numerics within the documented band, then emit one
signed receipt per backend naming the backend actually used.
