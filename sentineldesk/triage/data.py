"""Build/read split parquet files and enforce the leakage rules.

The 80/20 split itself is the team's `load_split` in the repo-root `split.py` (team rule:
one split, one place; never re-shuffle here). This module only:
  * makes sure `data/tickets.csv` exists (downloads the agreed HF file if it is missing),
  * cuts the triage calibration slices (half A / half B) out of kb,
  * writes parquet files and checks for leakage.
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sentineldesk.triage.config import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from split import load_split  # noqa: E402  (the team's locked split)

log = logging.getLogger(__name__)

DATASET_REPO = "Tobi-Bueck/customer-support-tickets"
DATASET_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
DEFAULT_CSV = REPO_ROOT / "data" / "tickets.csv"
SEED = 42
TRAIN_PARTS = ("train", "half_a", "half_b")  # cut from kb by paraphrase family in train_triage.py
TEST_INPUT_COLS = ["ticket_id", "subject", "body"]
TEST_LABEL_COLS = ["ticket_id", "type", "queue", "priority"]
# Columns kept for training parts. `answer` is kept for the retrieval teammate (kb only).
TRAIN_COLS = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority"] + [
    f"tag_{i}" for i in range(1, 9)
]


def ensure_csv(path: str | Path | None = None) -> Path:
    """Return the dataset CSV path; if it is missing, download the agreed HF file to it.
    If the data owner has put a different (e.g. deduped) file there, it is used as-is."""
    path = Path(path) if path else DEFAULT_CSV
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(DATASET_REPO, repo_type="dataset")
    matches = [f for f in files if Path(f).name == DATASET_FILE]
    if not matches:
        raise FileNotFoundError(f"{DATASET_FILE} not found in {DATASET_REPO}. Files in the repo: {files}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(DATASET_REPO, matches[0], repo_type="dataset"), path)
    log.info("downloaded %s -> %s", DATASET_FILE, path)
    return path


def sanity_check(kb: pd.DataFrame, test: pd.DataFrame) -> None:
    """Expect 4 types and 3 priorities (checked on kb only; holdout labels stay untouched)."""
    missing = {"subject", "body", "type", "queue", "priority"} - set(kb.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    log.info("kb=%d test=%d  missing subjects in kb=%d", len(kb), len(test), int(kb["subject"].isna().sum()))
    for col in ("type", "queue", "priority"):
        log.info("%s: %s", col, sorted(kb[col].dropna().unique().tolist()))
    n_type, n_prio = kb["type"].nunique(), kb["priority"].nunique()
    if n_type != 4 or n_prio != 3:
        raise ValueError(f"Expected 4 types and 3 priorities, got {n_type} and {n_prio}. Wrong dataset file?")


KB_INDEX_COLS = ["ticket_id", "text", "subject", "answer", "language", "type", "queue", "priority"] + [
    f"tag_{i}" for i in range(1, 9)
]
_PLACEHOLDER = re.compile(r"<[^<>\s]{1,40}>")


def is_searchable(answer) -> bool:
    """README rule: False for empty, placeholder-only or under-40-character answers."""
    if answer is None or (isinstance(answer, float) and math.isnan(answer)):
        return False
    a = str(answer).strip()
    return len(a) >= 40 and bool(re.search(r"[^\W\d_]", _PLACEHOLDER.sub("", a)))


def kb_index_frame(kb: pd.DataFrame) -> pd.DataFrame:
    """Rows for data/index/kb.pkl, in load_split() order, plus the `searchable` flag."""
    out = kb[[c for c in KB_INDEX_COLS if c in kb.columns]].reset_index(drop=True).copy()
    out["searchable"] = out["answer"].map(is_searchable)
    return out


def paraphrase_families(emb: np.ndarray, threshold: float = 0.98, chunk: int = 2048) -> np.ndarray:
    """One family id per row: tickets whose normalized embeddings have cosine >= threshold are linked
    (connected components). docs/DATA_PREPROCESSING.md, Appendix A; 0.98 re-checked on e5-base: labels
    agree ~97-100% above it and drop to ~79% (queue) just below."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols = [], []
    for start in range(0, len(emb), chunk):
        r, c = np.nonzero(emb[start:start + chunk] @ emb.T >= threshold)
        keep = r + start < c
        rows.append(r[keep] + start)
        cols.append(c[keep])
    r, c = (np.concatenate(rows), np.concatenate(cols)) if rows else (np.array([], int), np.array([], int))
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(emb), len(emb)))
    return connected_components(graph, directed=False)[1]


def family_calib_split(kb: pd.DataFrame, emb: np.ndarray, threshold: float = 0.98
                       ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """kb rows -> positions of train (~90%) / half_a (~5%) / half_b (~5%), drawn by WHOLE paraphrase
    families (seed 42), so no calibration ticket has a reworded copy in train (audit items 3 + 4).
    Returns (positions per part, family id per kb row)."""
    from sklearn.model_selection import GroupShuffleSplit

    if len(kb) != len(emb):
        raise ValueError("kb rows and embeddings are not aligned")
    fam = paraphrase_families(emb, threshold)
    tr, cal = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED).split(emb, groups=fam))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(cal, groups=fam[cal]))
    pos = {"train": np.sort(tr), "half_a": np.sort(cal[a]), "half_b": np.sort(cal[b])}
    for x, y in (("train", "half_a"), ("train", "half_b"), ("half_a", "half_b")):
        if set(fam[pos[x]]) & set(fam[pos[y]]):
            raise AssertionError(f"family shared between {x} and {y}")
    return pos, fam


def build_parts(csv_path: str | Path | None = None, sample: int | None = None) -> dict[str, pd.DataFrame]:
    """Team split via split.load_split -> {"kb", "test"}. The calibration halves are cut later, by
    paraphrase family, once kb embeddings exist (scripts/train_triage.py). In sample mode kb is
    subsampled so train ends up ~N tickets, test proportionally (seed 42)."""
    kb, test = load_split(str(ensure_csv(csv_path)))
    sanity_check(kb, test)
    parts = {"kb": kb, "test": test}
    if sample:
        frac = min(1.0, sample / (0.9 * len(kb)))
        parts = {k: v.sample(frac=frac, random_state=SEED).sort_index() for k, v in parts.items()}
    assert_disjoint({k: v["ticket_id"] for k, v in parts.items()})
    if "answer" in parts["test"].columns:
        raise AssertionError("LEAKAGE: holdout answers left split.py")
    return parts


def assert_disjoint(id_sets: dict[str, "pd.Series | list[str]"]) -> None:
    """Raise if any ticket_id appears in two parts (catches holdout leakage)."""
    seen: dict[str, str] = {}
    for name, ids in id_sets.items():
        for tid in ids:
            if tid in seen:
                raise AssertionError(f"LEAKAGE: {tid} is in both {seen[tid]} and {name}")
            seen[tid] = name


def assert_no_test_ids(ids, test_ids) -> None:
    overlap = set(ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"LEAKAGE: {len(overlap)} holdout ticket_ids found, e.g. {sorted(overlap)[:3]}")


def write_parts(parts: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """kb.parquet (training + knowledge base) and the holdout split into inputs / labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = parts["kb"]
    kb[[c for c in TRAIN_COLS if c in kb.columns]].to_parquet(out_dir / "kb.parquet", index=False)
    test = parts["test"]
    test[TEST_INPUT_COLS].to_parquet(out_dir / "test_inputs.parquet", index=False)
    test[TEST_LABEL_COLS].to_parquet(out_dir / "test_labels.parquet", index=False)
    (out_dir / "sizes.json").write_text(json.dumps({k: len(v) for k, v in parts.items()}, indent=2))
    log.info("wrote splits to %s: %s", out_dir, {k: len(v) for k, v in parts.items()})


def write_calib_parts(kb: pd.DataFrame, pos: dict[str, np.ndarray], fam: np.ndarray, out_dir: Path) -> None:
    """Record the family-based train / half_a / half_b actually used for training (for eval + audit)."""
    cols = [c for c in TRAIN_COLS if c in kb.columns]
    for name, p in pos.items():
        kb.iloc[p][cols].to_parquet(out_dir / f"{name}.parquet", index=False)
    pd.DataFrame({"ticket_id": kb["ticket_id"].values, "family": fam}).to_parquet(out_dir / "families.parquet", index=False)


def read_part(splits_dir: Path, name: str) -> pd.DataFrame:
    """Read kb/train/half_a/half_b/test_inputs. test_labels is only read by eval, via read_test_labels."""
    if name == "test_labels":
        raise PermissionError("Use read_test_labels() in scripts/eval_triage.py, after predictions are saved")
    return pd.read_parquet(splits_dir / f"{name}.parquet")


def read_test_labels(splits_dir: Path, predictions_path: Path) -> pd.DataFrame:
    """Only for evaluation: refuses to run until holdout predictions exist on disk."""
    if not predictions_path.exists():
        raise PermissionError(f"Holdout labels are locked until predictions are saved to {predictions_path}")
    return pd.read_parquet(splits_dir / "test_labels.parquet")
