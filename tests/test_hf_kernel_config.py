# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""This repository's hf-kernel-publish config and cards pass the publisher's checks.

Offline; no Hub access. The workflow's build job enforces the same rules on the
committed source, but only on path-filtered pull requests. These tests fail on any
pull request that breaks the card lint (CARD.md is rendered from hf/card.yaml) or the
config, before a release would.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ID = "SZLHOLDINGS/szl-receipt-attn"


def _publisher():
    name = "szl_kernel_publish_repo_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "szl_kernel_publish.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_config_names_this_kernel_and_package():
    kp = _publisher()
    cfg = kp.find_kernel(_ROOT, _REPO_ID)
    assert cfg.name == "szl-receipt-attn"
    assert cfg.version == 1
    assert cfg.source_repository == "szl-holdings/szl-receipt-attn"
    assert [(p.src, p.name) for p in cfg.packages] == [("torch-ext/szl_receipt_attn", "szl_receipt_attn")]
    assert (_ROOT / cfg.packages[0].src / "__init__.py").is_file()
    assert cfg.card == "CARD.md" and cfg.license_file == "LICENSE"
    assert cfg.twin is not None and {"build/", "torch-ext/"} <= set(cfg.twin.prune_paths)


def test_kernel_card_and_pointer_pass_the_publisher_lint():
    kp = _publisher()
    cfg = kp.find_kernel(_ROOT, _REPO_ID)
    for rel in (cfg.card, cfg.twin.card):
        text = kp.normalize_text(rel, (_ROOT / rel).read_bytes()).decode("utf-8")
        rendered = kp.render_card(cfg, text, rel)
        kp.lint_card(cfg, rendered, rel, require_usage=True)
        assert f'get_kernel("{_REPO_ID}", version=1, trust_remote_code=True)' in rendered


def test_package_sources_are_publishable_text():
    kp = _publisher()
    cfg = kp.find_kernel(_ROOT, _REPO_ID)
    files = [p for p in (_ROOT / cfg.packages[0].src).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    assert files
    for path in files:
        kp.normalize_text(path.relative_to(_ROOT).as_posix(), path.read_bytes())
