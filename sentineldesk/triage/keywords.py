"""Outage keyword list -> features for the critical detector. Never used to change labels."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from sentineldesk.triage.text import normalize

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def _canon(text: str) -> str:
    return normalize(text).translate(_APOSTROPHES).lower()


class KeywordMatcher:
    """Unicode-aware, word-boundary phrase matching ("down" never matches "download")."""

    def __init__(self, keywords: list[str]):
        self.keywords = sorted({_canon(k).strip() for k in keywords if k and k.strip()})
        self._patterns = [
            (kw, re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, kw.split())) + r"(?!\w)"))
            for kw in self.keywords
        ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KeywordMatcher":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls([kw for lang_list in data.values() for kw in (lang_list or [])])

    def matches(self, text: str) -> tuple[list[str], int]:
        """Return (distinct keywords hit, total number of hits)."""
        t = _canon(text)
        hits, total = [], 0
        for kw, rx in self._patterns:
            n = len(rx.findall(t))
            if n:
                hits.append(kw)
                total += n
        return hits, total

    def features(self, texts: list[str]) -> tuple[np.ndarray, list[list[str]]]:
        """(n, 3) array [any_hit, n_hits, n_distinct_hits] + list of hits per text."""
        feats, all_hits = np.zeros((len(texts), 3), dtype=np.float32), []
        for i, t in enumerate(texts):
            hits, total = self.matches(t)
            feats[i] = (1.0 if hits else 0.0, total, len(hits))
            all_hits.append(hits)
        return feats, all_hits
