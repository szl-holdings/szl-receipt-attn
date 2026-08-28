# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Original SHA3-256 attention-call receipt helper for szl-receipt-attn.

This module is written for this package. It is not a copy of
szl-kernels `_chain.py`. The digest is an integrity fingerprint of
canonical JSON, not a signature and not proven trust.

Λ uniqueness remains Conjecture 1 (OPEN) / advisory.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

import torch

from .const import LAMBDA_STATUS, PACKAGE_NAME

GENESIS = f"{PACKAGE_NAME}:genesis:v1"


def _canonical_bytes(body: Mapping[str, Any]) -> bytes:
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def sha3_256_hex(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return cpu.numpy().tobytes()


def digest_mask(attn_mask: Optional[torch.Tensor]) -> str:
    """SHA3-256 over mask shape, dtype, and contiguous payload bytes."""
    if attn_mask is None:
        return "none"
    payload = {
        "dtype": str(attn_mask.dtype).replace("torch.", ""),
        "shape": list(attn_mask.shape),
        "sha3_256": sha3_256_hex(_tensor_raw_bytes(attn_mask)),
    }
    return sha3_256_hex(_canonical_bytes(payload))


def shape_list(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


class UnifiedReceiptChain:
    """Hash-chained attention receipts (SHA3-256).

    ``verify()`` never fabricates a pass: a broken link stays failed.
    """

    def __init__(self) -> None:
        self._receipts: list[dict[str, Any]] = []
        self._head = sha3_256_hex(GENESIS.encode("utf-8"))

    @property
    def head(self) -> str:
        return self._head

    def __len__(self) -> int:
        return len(self._receipts)

    def receipts(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._receipts)

    def emit(self, body: Mapping[str, Any]) -> str:
        record = {
            "body": dict(body),
            "lambda": LAMBDA_STATUS,
            "package": PACKAGE_NAME,
            "prev": self._head,
        }
        digest = sha3_256_hex(_canonical_bytes(record))
        stored = dict(record)
        stored["digest"] = digest
        self._receipts.append(stored)
        self._head = digest
        return digest

    def verify(self) -> tuple[bool, int, Optional[int]]:
        """Return ``(ok, depth, first_break_index)``.

        ``ok`` is True only when every stored link re-hashes. A tampered
        or truncated chain stays failed.
        """
        prev = sha3_256_hex(GENESIS.encode("utf-8"))
        for index, stored in enumerate(self._receipts):
            expected_prev = stored.get("prev")
            if expected_prev != prev:
                return False, index, index
            replay = {
                "body": stored.get("body"),
                "lambda": stored.get("lambda"),
                "package": stored.get("package"),
                "prev": stored.get("prev"),
            }
            recomputed = sha3_256_hex(_canonical_bytes(replay))
            if recomputed != stored.get("digest"):
                return False, index, index
            prev = recomputed
        return True, len(self._receipts), None

    def to_json(self) -> str:
        return json.dumps(
            {
                "genesis": GENESIS,
                "head": self._head,
                "receipts": self._receipts,
            },
            sort_keys=True,
            indent=2,
        )

    @classmethod
    def from_json(cls, blob: str) -> "UnifiedReceiptChain":
        data = json.loads(blob)
        chain = cls()
        chain._receipts = list(data.get("receipts") or [])
        chain._head = data.get("head") or chain._head
        return chain
