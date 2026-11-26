"""Stand-in for retrieve.search() (Chirag; locked interface in README) until it is pushed.

Dense cosine only over the shared index (data/index/, built by scripts/embed.py). Same signature and return
shape as the locked search(), plus three optional extras the console shows when present:
`body`, `tags` and `answer_language`. No BM25 / RRF / boosts: `preds` and `lang` are accepted and ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentineldesk.language import detect_language

TAG_COLS = [f"tag_{i}" for i in range(1, 9)]


def _str(v) -> str:
    return v if isinstance(v, str) else ""          # ~16% of KB subjects are missing (NaN)


class DenseSearch:
    def __init__(self, index_dir: Path, encode):
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.E = np.load(index_dir / "emb.npy")
        self.kb = pd.read_pickle(index_dir / "kb.pkl").reset_index(drop=True)
        self.searchable = self.kb["searchable"].to_numpy(dtype=bool)   # junk answers are never returned
        self.row = {tid: i for i, tid in enumerate(self.kb["ticket_id"])}
        # The dataset's language label is wrong for ~27% of "de" rows (docs/DATA_PREPROCESSING.md), so the
        # console shows the language detected from the text (~1 s for the whole KB, once at start-up).
        self.lang = [detect_language("", t) for t in self.kb["text"]]
        self.answer_lang = [detect_language("", _str(a)) for a in self.kb["answer"]]
        self.encode = encode

    def search(self, text, k=5, preds=None, lang=None, exclude_ids=None, pool_ids=None) -> list[dict]:
        s = self.E @ self.encode([text])[0]
        allowed = self.searchable.copy()
        if pool_ids is not None:
            pool = np.zeros_like(allowed)
            pool[[self.row[t] for t in pool_ids if t in self.row]] = True
            allowed &= pool
        for t in exclude_ids or ():
            if t in self.row:
                allowed[self.row[t]] = False
        s = np.where(allowed, s, -np.inf)
        cand = np.argpartition(-s, min(len(s) - 1, 4 * k))[: 4 * k]
        cand = cand[np.argsort(-s[cand])]
        out, seen = [], set()
        for i in map(int, cand):
            r = self.kb.iloc[i]
            key = " ".join(_str(r["answer"]).split()).lower()
            if key in seen:                  # README: drop near-duplicate answers (exact match here)
                continue
            seen.add(key)
            subject = _str(r["subject"])
            body = r["text"][len(subject) + 1:] if r["text"].startswith(subject + "\n") else r["text"]
            out.append({
                "ticket_id": str(r["ticket_id"]), "similarity": round(float(s[i]), 4), "language": self.lang[i],
                "type": r["type"], "queue": r["queue"], "priority": r["priority"],
                "subject": subject, "answer": _str(r["answer"]),
                "body": body, "answer_language": self.answer_lang[i],
                "tags": [t for t in (r[c] for c in TAG_COLS) if isinstance(t, str) and t],
            })
            if len(out) == k:
                break
        return out
