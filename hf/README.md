# Hugging Face publication

This folder is the GitHub source for the two Hub twins of this repository:

- the Kernel Hub repo [`kernels/SZLHOLDINGS/szl-receipt-attn`](https://huggingface.co/kernels/SZLHOLDINGS/szl-receipt-attn), which is what `get_kernel` loads;
- the model-type twin [`SZLHOLDINGS/szl-receipt-attn`](https://huggingface.co/SZLHOLDINGS/szl-receipt-attn), which is a pointer to the kernel repo and holds no loadable build.

```python
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/szl-receipt-attn", version=1, trust_remote_code=True)
```

| Path | What it is |
| --- | --- |
| `card.yaml` | The kernel card source. |
| `../CARD.md` | Its rendering by the shared [`hf-card`](https://github.com/szl-holdings/.github/tree/main/hf-card) toolkit. `hf-kernel-publish` publishes it as the kernel repo `README.md`. |
| `model-twin-pointer.md` | The model twin's `README.md`: a pointer to the kernel repo. |
| `kernel_smoke.py` | The smoke hook `hf-kernel-publish` runs on every loaded build (local and Hub): `selfcheck()`, a CPU comparison with torch SDPA, and one receipt that must verify. |
| `hub-import/` | Historical snapshot: the Hub cards and file lists of both twins as the Hub served them on 2026-09-29, before the canonical publisher existed. Kept as a record; it is not refreshed by publication. |

## How the Hub is written

`.github/workflows/hf-kernel-publish.yml` with `scripts/szl_kernel_publish.py` and the
config `.github/hf-kernels.json` is the only writer of both twins.

- **Pull request:** builds `build/torch-cpu` and `build/torch-universal` from
  `torch-ext/szl_receipt_attn` and loads both with `kernels==0.17.2` and CPU torch. No Hub
  access, no secret.
- **Release published** (or a dispatch from `main`): the same build, then a fast-forward Git
  push of one managed tree to the kernel repo's `main` and `v1` branches. The publisher owns
  all of `build/`, so superseded modules are deleted. It also writes `CARD.md` as `README.md`,
  `LICENSE`, and `SZL_KERNEL_SOURCE.json` (the GitHub commit the build came from), and prunes
  the stale Hub-only plates `OPERATIONAL.json` and `BENCH.laptop-blackwell.json`.
- **Verify:** a fresh, anonymous job calls `get_kernel(..., version=1)` and
  `get_kernel(..., revision=...)` on `main`, and fails if the served files differ from GitHub.
- **Model twin:** only after verify passes, its `README.md` becomes the pointer card, and
  `build/`, `torch-ext/`, `tests/`, `build.toml`, the stray `receipt_attn.py` upload and the
  stale plates are removed. `LICENSE`, `bom/` and `og-card.png` are kept.
- **Weekly:** a drift guard re-verifies what `get_kernel` serves and fails with `SOURCE_AHEAD`
  when the latest release changed the kernel but was never published.

The kernel major is `version` in `.github/hf-kernels.json`, not the release tag. Bumping it
starts `v2` and leaves `v1` frozen. The credential is the org secret `HF_TOKEN`, exposed only
to the two writing steps.

To repair or republish, dispatch `hf-kernel-publish` from `main` with `source_ref` (a tag,
branch or commit on `main`); `dry_run=true` plans and checks the write without pushing.

CI also keeps the card source honest:

- `.github/workflows/hf-card.yml` fails when `CARD.md` drifts from `card.yaml`, or when the
  card breaks the shared card contract.
- `hf-kernel-publish` lints the card it publishes: no floating `main` revision, no bare
  `get_kernel` call, no placeholders, and `version=1` must be shown.
- `tests/test_hf_card_source.py` checks, offline, that `hub-import/` still holds the imported
  bytes, and that `CARD.md` names this repository and its Apache-2.0 license.
- `tests/test_hf_kernel_config.py` checks, offline, that the publisher config, the kernel card
  and the pointer card pass the publisher's own validation.

To edit the card, change `card.yaml` and re-render:

```bash
python <hf-card>/render.py hf/card.yaml --out CARD.md
```

## Other writers

The `szl-holdings/a11oy` `atelier-hub-publish.yml` workflow has uploaded `receipt_attn.py` to
the model twin. That cross-repository writer is to be retired; until then the twin publish
removes that file again.
