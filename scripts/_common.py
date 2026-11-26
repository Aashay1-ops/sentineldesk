"""Shared CLI plumbing for the triage scripts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to triage.yaml (default configs/triage.yaml)")
    p.add_argument("--artifacts", default=None, help="artifacts root (overrides config / SENTINELDESK_ARTIFACTS)")
    p.add_argument("--sample", type=int, default=None, help="quick run on N training tickets (separate artifacts dir)")
    return p


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
