# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""TK-style tile-level portability reference for szl-receipt-attn.

OUROBOROS round-2 leader pack, Cluster C absorb: ThunderKittens-style
tile abstraction as the *portability reference* for the receipt-attn
consumer-GPU branch. Acceptance test: one kernel schedule expressed at
tile level, two reference backends, identical numeric tolerance, one
hash-chained receipt per backend.

Honesty: this module runs on host CPU with Python floats (fp64).
No GPU timing, no Triton cubin, no speed claims and no fabricated
benchmarks are produced here. Receipts are fail-closed: a mismatched
tolerance yields verdict FAIL and a nonzero exit code.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone

SCHEMA = "szl.tkport/v1"
NEG_INF = float("-inf")
ATOL = 1e-9
DEFAULT_SEED = 20261126


def make_inputs(heads=2, seq=16, dim=32, seed=DEFAULT_SEED):
    """Deterministic fp64 inputs shaped (heads, seq, dim)."""
    rng = random.Random(seed)

    def one():
        return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(seq)]

    q = [one() for _ in range(heads)]
    k = [one() for _ in range(heads)]
    v = [one() for _ in range(heads)]
    return q, k, v


def _scale(dim: int) -> float:
    return 1.0 / math.sqrt(dim)


def _row_dot(a, b, scale):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s * scale


def eager_attention_head(q, k, v, causal: bool):
    """Backend 2: full-matrix reference (one head)."""
    seq, dim = len(q), len(q[0])
    scale = _scale(dim)
    out = []
    for i in range(seq):
        scores = []
        for j in range(seq):
            if causal and j > i:
                scores.append(NEG_INF)
            else:
                scores.append(_row_dot(q[i], k[j], scale))
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        denom = sum(exps)
        row = [0.0] * dim
        for j, p in enumerate(exps):
            if p == 0.0:
                continue
            vj = v[j]
            for t in range(dim):
                row[t] += p * vj[t]
        out.append([x / denom for x in row])
    return out


def tiled_attention_head(q, k, v, causal: bool, block_q=4, block_kv=4):
    """Backend 1: flash-style online-softmax tile schedule (one head).

    Tile abstraction mirrors the Triton schedule in
    torch-ext/szl_receipt_attn: iterate Q tiles, stream K/V tiles,
    maintain per-row running max / normalizer / accumulator, rescale on
    every tile. Backend-neutral by construction: only tile math, no
    memory layout or backend intrinsics are expressed here.
    """
    seq, dim = len(q), len(q[0])
    scale = _scale(dim)
    out = []
    for qs in range(0, seq, block_q):
        bq = min(block_q, seq - qs)
        m_row = [NEG_INF] * bq
        l_row = [0.0] * bq
        acc = [[0.0] * dim for _ in range(bq)]
        for ks in range(0, seq, block_kv):
            bkv = min(block_kv, seq - ks)
            for i in range(bq):
                qi = qs + i
                tile_scores = []
                for j in range(bkv):
                    kj = ks + j
                    if causal and kj > qi:
                        tile_scores.append(NEG_INF)
                    else:
                        tile_scores.append(_row_dot(q[qi], k[kj], scale))
                m_new = max(m_row[i], max(tile_scores))
                if m_new == NEG_INF:
                    continue
                corr = 0.0 if m_row[i] == NEG_INF else math.exp(m_row[i] - m_new)
                p = [math.exp(s - m_new) for s in tile_scores]
                l_row[i] = l_row[i] * corr + sum(p)
                acc[i] = [x * corr for x in acc[i]]
                for j, pj in enumerate(p):
                    if pj == 0.0:
                        continue
                    vj = v[ks + j]
                    ai = acc[i]
                    for t in range(dim):
                        ai[t] += pj * vj[t]
                m_row[i] = m_new
        for i in range(bq):
            out.append([x / l_row[i] for x in acc[i]] if l_row[i] > 0.0 else [0.0] * dim)
    return out


def attention(qh, kh, vh, causal: bool, backend: str, block_q=4, block_kv=4):
    heads = []
    for q, k, v in zip(qh, kh, vh):
        if backend == "python_tiled_flash":
            heads.append(tiled_attention_head(q, k, v, causal, block_q, block_kv))
        elif backend == "python_eager":
            heads.append(eager_attention_head(q, k, v, causal))
        else:
            raise ValueError("unknown backend: " + backend)
    return heads


def max_abs_diff(a, b):
    m = 0.0
    for ha, hb in zip(a, b):
        for ra, rb in zip(ha, hb):
            for xa, xb in zip(ra, rb):
                d = abs(xa - xb)
                if d > m:
                    m = d
    return m


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def receipt_id(receipt: dict) -> str:
    payload = {k: v for k, v in receipt.items() if k != "sha256"}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def run_conformance(causal=True, seed=DEFAULT_SEED):
    """Both backends, one comparison, one receipt per backend, chained."""
    q, k, v = make_inputs(seed=seed)
    shapes = {"heads": len(q), "seq": len(q[0]), "dim": len(q[0][0]), "causal": causal, "seed": seed}
    out_tiled = attention(q, k, v, causal, "python_tiled_flash")
    out_eager = attention(q, k, v, causal, "python_eager")
    diff = max_abs_diff(out_tiled, out_eager)
    verdict = "PASS" if diff <= ATOL else "FAIL"
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "schema": SCHEMA,
        "kernel": "szl-receipt-attn tile schedule (portability reference)",
        "shapes": shapes,
        "comparison": {"atol_tiled_vs_eager": ATOL, "max_abs_diff": diff},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mode": "host-cpu fp64 pure-stdlib",
            "about_timing": "wall-clock not recorded; this is not a benchmark",
        },
        "utc": run_utc,
        "what_not": [
            "no GPU execution",
            "no Triton cubin",
            "not a speed benchmark and no speedup claim",
            "not a Hub publication and not a Hub-load qualification",
        ],
    }
    receipts = []
    for backend in ("python_tiled_flash", "python_eager"):
        r = dict(base)
        r["backend"] = backend
        r["verdict"] = verdict
        if receipts:
            r["chain_prev_sha256"] = receipts[-1]["sha256"]
        r["sha256"] = receipt_id(r)
        receipts.append(r)
    return {"schema": SCHEMA, "verdict": verdict, "atol": ATOL, "max_abs_diff": diff, "receipts": receipts}


def main(argv):
    out_path = argv[1] if len(argv) > 1 else "tk_portability_receipts.json"
    result = run_conformance()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("TKCONFORMANCE_VERDICT " + result["verdict"] + " max_abs_diff=" + repr(result["max_abs_diff"]) + " atol=" + repr(ATOL))
    for r in result["receipts"]:
        print("RECEIPT " + r["backend"] + " " + r["sha256"])
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
