"""SentinelDesk: Hybrid customer support triage, retrieval, safety gate, and grounded drafting."""

from sentineldesk.language import (
    BUILTIN_PROFILES,
    DEFAULT_DETECTOR,
    DE_PROFILE,
    EN_PROFILE,
    LanguageDetector,
    LanguagePrediction,
    LanguageProfile,
    annotate_dataframe_language,
    detect_language,
    detect_language_batch,
    detect_language_with_confidence,
)

__all__ = [
    "BUILTIN_PROFILES",
    "DEFAULT_DETECTOR",
    "DE_PROFILE",
    "EN_PROFILE",
    "LanguageDetector",
    "LanguagePrediction",
    "LanguageProfile",
    "annotate_dataframe_language",
    "detect_language",
    "detect_language_batch",
    "detect_language_with_confidence",
]
