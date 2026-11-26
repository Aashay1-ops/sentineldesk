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
TRAIN_PARTS = ("train", "half_a", "half_b")
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
_WORD = re.compile(r"(?u)\b[^\W\d_]{2,}\b")
# Data audit #7 (docs/DATA_PREPROCESSING.md section 5): a customer body written in the agent's voice.
_AGENT_OPENER = re.compile(
    r"(?i)^\s*(?:thank you|thanks for|we appreciate|we apologi[sz]e|we are sorry|we're sorry|"
    r"vielen dank|danke für|wir entschuldigen|es tut uns leid|wir freuen uns|wir schätzen|wir danken|"
    r"(?:dear|hello|hi|hallo|sehr geehrte\w*)\s+(?:\[|<|\{)?\s*(?:name|ihr name)\b)")


def _text(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def is_searchable(answer, subject=None, body=None) -> bool:
    """False for empty, placeholder-only or under-40-character answers (README), and, when subject/body
    are given, for the audit's broken rows: agent-voice opener, body/answer word Jaccard >= 0.60
    (copied or echo-only answer), or under 6 words of subject + body.
    ponytail: the audit's agent-voice classifier is not used (needs a trained model); ~2.5% of
    broken rows remain per the audit's sample."""
    a = _text(answer).strip()
    if len(a) < 40 or not re.search(r"[^\W\d_]", _PLACEHOLDER.sub("", a)):
        return False
    if body is None:
        return True
    s, b = _text(subject), _text(body)
    if len((s + " " + b).split()) < 6 or _AGENT_OPENER.match(b):
        return False
    wa, wb = {w.lower() for w in _WORD.findall(a)}, {w.lower() for w in _WORD.findall(b)}
    return not (wa and wb and len(wa & wb) / len(wa | wb) >= 0.60)


def kb_index_frame(kb: pd.DataFrame) -> pd.DataFrame:
    """Rows for data/index/kb.pkl, in load_split() order, plus the `searchable` flag."""
    out = kb[[c for c in KB_INDEX_COLS if c in kb.columns]].reset_index(drop=True).copy()
    body = kb["body"].reset_index(drop=True) if "body" in kb.columns else [None] * len(out)
    out["searchable"] = [is_searchable(a, s, b) for a, s, b in zip(out["answer"], out["subject"], body)]
    return out


def calib_split(kb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """kb -> (train, half_a, half_b): 90% / 5% / 5%, seed 42."""
    calib = kb.sample(frac=0.10, random_state=SEED)
    train = kb.drop(calib.index)
    half_a = calib.sample(frac=0.5, random_state=SEED)
    half_b = calib.drop(half_a.index)
    return train, half_a, half_b


def paraphrase_families(emb, threshold: float = 0.98, chunk: int = 2048):
    """One family id per row: rows whose normalized embeddings have cosine >= threshold are
    connected (data audit, Appendix A). Returns (family ids, top-1 neighbour similarity, top-1 neighbour row)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(emb)
    rows, cols = [], []
    top1, nn1 = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        S = emb[s:s + chunk] @ emb.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -1.0  # ignore self
        nn1[s:s + len(S)] = S.argmax(axis=1)
        top1[s:s + len(S)] = S[np.arange(len(S)), nn1[s:s + len(S)]]
        r, c = np.nonzero(S >= threshold)
        keep = r + s < c
        rows.append(r[keep] + s)
        cols.append(c[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(graph, directed=False)[1], top1, nn1


def family_calib_split(kb: pd.DataFrame, family) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like calib_split (~90/5/5 of tickets: the 10%/50% cuts are of families, seed 42), but whole families stay on one side, so
    calibration, ensemble weights and the critical threshold are tuned on tickets that have no
    reworded copy in train (data audit #3; also fixes #4, because neighbour votes search train only)."""
    from sklearn.model_selection import GroupShuffleSplit

    kb = kb.reset_index(drop=True)
    tr, cal = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED).split(kb, groups=family))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(cal, groups=family[cal]))
    train, half_a, half_b = kb.iloc[tr], kb.iloc[cal[a]], kb.iloc[cal[b]]
    for x, y in ((train, half_a), (train, half_b), (half_a, half_b)):
        shared = set(family[x.index]) & set(family[y.index])
        if shared:
            raise AssertionError(f"{len(shared)} paraphrase families span two calibration parts")
    return train, half_a, half_b


def build_parts(csv_path: str | Path | None = None, sample: int | None = None) -> dict[str, pd.DataFrame]:
    """Team split via split.load_split, then triage calibration slices; in sample mode,
    subsample every part proportionally (seed 42)."""
    kb, test = load_split(str(ensure_csv(csv_path)))
    sanity_check(kb, test)
    train, half_a, half_b = calib_split(kb)
    parts = {"train": train, "half_a": half_a, "half_b": half_b, "test": test}
    if sample:
        frac = min(1.0, sample / len(train))
        parts = {k: v.sample(frac=frac, random_state=SEED) for k, v in parts.items()}
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
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_PARTS:
        cols = [c for c in TRAIN_COLS if c in parts[name].columns]
        parts[name][cols].to_parquet(out_dir / f"{name}.parquet", index=False)
    test = parts["test"]
    test[TEST_INPUT_COLS].to_parquet(out_dir / "test_inputs.parquet", index=False)
    test[TEST_LABEL_COLS].to_parquet(out_dir / "test_labels.parquet", index=False)
    (out_dir / "sizes.json").write_text(json.dumps({k: len(v) for k, v in parts.items()}, indent=2))
    log.info("wrote splits to %s: %s", out_dir, {k: len(v) for k, v in parts.items()})


def read_part(splits_dir: Path, name: str) -> pd.DataFrame:
    """Read train/half_a/half_b/test_inputs. test_labels is only read by eval, via read_test_labels."""
    if name == "test_labels":
        raise PermissionError("Use read_test_labels() in scripts/eval_triage.py, after predictions are saved")
    return pd.read_parquet(splits_dir / f"{name}.parquet")


def read_test_labels(splits_dir: Path, predictions_path: Path) -> pd.DataFrame:
    """Only for evaluation: refuses to run until holdout predictions exist on disk."""
    if not predictions_path.exists():
        raise PermissionError(f"Holdout labels are locked until predictions are saved to {predictions_path}")
    return pd.read_parquet(splits_dir / "test_labels.parquet")
