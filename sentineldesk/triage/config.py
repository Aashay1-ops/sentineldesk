"""Load configs/triage.yaml and resolve artifact paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "triage.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def artifacts_dir(cfg: dict, override: str | Path | None = None, sample: int | None = None) -> Path:
    """Artifacts root: CLI override > SENTINELDESK_ARTIFACTS env > config. Sample runs get a subfolder
    so they never overwrite full-size artifacts."""
    root = Path(override or os.environ.get("SENTINELDESK_ARTIFACTS") or cfg["paths"]["artifacts_dir"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / f"sample_{sample}" if sample else root


def results_dir(cfg: dict, sample: int | None = None) -> Path:
    root = Path(cfg["paths"]["results_dir"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / f"sample_{sample}" if sample else root


def embedding_prefix(cfg: dict, model_name: str) -> str:
    return cfg["embedding"].get("prefixes", {}).get(model_name, "")


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")
