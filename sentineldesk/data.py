"""Dataset loading and the locked 80/20 split (exactly as in the official quick-start)."""
from __future__ import annotations

import pandas as pd

from . import config

TARGETS = ["type", "queue", "priority"]


def _download() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=config.HF_REPO, filename=config.HF_FILE, repo_type="dataset")
    return pd.read_csv(path)


def build_split(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle (seed 42) -> assign TKT ids -> first 80% = KB/train, last 20% = holdout. Cached as parquet."""
    config.DATA_DIR.mkdir(exist_ok=True)
    kb_p, test_p = config.DATA_DIR / "kb_train.parquet", config.DATA_DIR / "test.parquet"
    if kb_p.exists() and test_p.exists() and not force:
        return pd.read_parquet(kb_p), pd.read_parquet(test_p)

    df = _download()
    df = df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    for col in ["subject", "body", "answer"]:
        # Some rows store newlines as literal "\n" / "<br>" sequences; normalise them.
        df[col] = (df[col].fillna("").astype(str)
                   .str.replace(r"\\n", "\n", regex=True).str.replace(r"<br\s*/?>", "\n", regex=True))
    split_idx = int(len(df) * config.KB_FRACTION)
    kb, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    kb.to_parquet(kb_p, index=False)
    test.to_parquet(test_p, index=False)
    return kb, test


def load_kb() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "kb_train.parquet")


def load_test_inputs_only() -> pd.DataFrame:
    """Holdout with ONLY inference-time inputs. Labels are loaded separately by evaluate.py for scoring."""
    return pd.read_parquet(config.DATA_DIR / "test.parquet", columns=["ticket_id", "subject", "body"])


def load_test_labels() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "test.parquet", columns=["ticket_id", "language", *TARGETS])


def ticket_text(subject: str | None, body: str | None) -> str:
    """Null-safe text join; missing subjects are common in the data."""
    s = (subject or "").strip()
    b = (body or "").strip()
    return f"{s}\n{b}" if s else b
