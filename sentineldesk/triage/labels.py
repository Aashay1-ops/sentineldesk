"""Fixed label order per task (sorted unique labels in `train`) and probability alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentineldesk.triage import TASKS


def make_labels(train: pd.DataFrame) -> dict[str, list[str]]:
    return {t: sorted(train[t].dropna().unique().tolist()) for t in TASKS}


def save_labels(labels: dict, path: Path) -> None:
    path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


def load_labels(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def encode(values, labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    return np.array([idx[v] for v in values], dtype=np.int64)


def align_proba(clf, probs: np.ndarray, labels: list[str]) -> np.ndarray:
    """Reorder sklearn predict_proba columns (clf.classes_) to `labels` order; missing classes get 0."""
    out = np.zeros((probs.shape[0], len(labels)), dtype=np.float64)
    pos = {l: i for i, l in enumerate(labels)}
    for j, c in enumerate(clf.classes_):
        out[:, pos[c]] = probs[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s == 0, 1, s)
