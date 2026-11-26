"""The one text builder used by training, evaluation and the live demo.

It reproduces split.py's `text` column EXACTLY (subject + "\\n" + body, missing -> ""), because the
team's shared embedding index (data/index/emb.npy) is built from that column: live tickets must be
embedded from the same string or they silently stop matching the index. Unicode normalization
(NFKC) is applied where it matters instead: keyword matching (keywords.py).
"""
from __future__ import annotations

import math
import re
import unicodedata

_BR = re.compile(r"(?i)<br\s*/?>")


def _clean(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    # same fix, same order as split.py (literal backslash-n, then <br>), so live text == training text
    return _BR.sub("\n", str(x).replace("\\n", "\n"))


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def build_text(subject, body) -> str:
    """subject (may be None/NaN/empty) + "\\n" + body, identical to split.py's `text`. Never "nan"."""
    s, b = _clean(subject), _clean(body)
    if not (s.strip() or b.strip()):
        raise ValueError("Ticket has no subject and no body")
    return s + "\n" + b


def build_texts(subjects, bodies) -> list[str]:
    return [build_text(s, b) for s, b in zip(subjects, bodies)]
