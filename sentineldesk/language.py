"""Language detection for tickets and replies (en / de), from the text only; never from a dataset column."""
from __future__ import annotations

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # deterministic results

_DE_HINTS = (" ich ", " der ", " die ", " und ", " nicht ", " bitte ", " sie ", " wir ", " ist ")


@dataclass
class LanguagePrediction:
    language: str
    confidence: float


def detect_language_with_confidence(subject: object = "", body: object = "") -> LanguagePrediction:
    text = f"{subject or ''}\n{body or ''}".strip()
    if not text:
        return LanguagePrediction("unknown", 0.0)
    try:
        best = detect_langs(text)[0]
        lang, conf = best.lang, float(best.prob)
    except LangDetectException:
        return LanguagePrediction("unknown", 0.0)
    if lang not in ("en", "de") and conf < 0.9:  # short texts confuse langdetect: fall back to en/de
        lang = "de" if any(h in f" {text.lower()} " for h in _DE_HINTS) else "en"
    return LanguagePrediction(lang, conf)


def detect_language(subject: object = "", body: object = "") -> str:
    return detect_language_with_confidence(subject, body).language
