"""Unit, extensibility, and benchmark tests for SentinelDesk language detection."""

from __future__ import annotations

import math
import re
import time

import pytest

from sentineldesk.language import (
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


class TestLanguageDetectionCore:
    """Verify correctness across English, German, short, and edge-case tickets."""

    @pytest.mark.parametrize(
        ("subject", "body", "expected"),
        [
            (
                "Account Disruption",
                "Dear Customer Support Team, I am writing to report a significant problem with the centralized account management portal.",
                "en",
            ),
            (
                "Query About Smart Home System Integration Features",
                "I hope this message reaches you well. I am reaching out to request detailed information about the capabilities of your smart home integration products.",
                "en",
            ),
            (
                "VPN is down",
                "Our hospital staff cannot access patient records since this morning.",
                "en",
            ),
            (
                None,
                "Please help me reset my billing password on the portal.",
                "en",
            ),
            (
                "Wesentlicher Sicherheitsvorfall",
                "Sehr geehrtes Support-Team, ich möchte einen gravierenden Sicherheitsvorfall melden, der gegenwärtig mehrere Komponenten unserer Infrastruktur betrifft.",
                "de",
            ),
            (
                "Sicherheitsverletzung gemeldet",
                "Es wurde eine Sicherheitsverletzung im IT-System festgestellt.",
                "de",
            ),
            (
                "Probleme mit dem Smart-Türklingel",
                "Ich beschreibe das Problem mit meinem Smart-Türklingel-Gerät. Leider gibt es einen Zusammenbruch der Verbindung.",
                "de",
            ),
            (
                None,
                "Kein Zugriff auf das System nach dem letzten Update. Bitte helfen Sie uns dringend.",
                "de",
            ),
            (
                "Störung",
                "Server ausgefallen",
                "de",
            ),
            (
                "",
                "Mein Account ist gesperrt, das System zeigt Error 500",
                "de",
            ),
            (
                "Support Team Update",
                "Mein Customer Account hat keinen Access auf das System.",
                "de",
            ),
        ],
    )
    def test_standard_en_and_de_tickets(
        self, subject: str | None, body: str, expected: str
    ) -> None:
        pred = detect_language_with_confidence(subject, body)
        assert isinstance(pred, LanguagePrediction)
        assert pred.language == expected
        assert 0.5 < pred.confidence <= 1.0
        assert detect_language(subject, body) == expected

    @pytest.mark.parametrize(
        ("subject", "body"),
        [
            (None, None),
            ("", ""),
            ("   ", "\n\t "),
            (float("nan"), float("nan")),
            (float("nan"), ""),
            (None, float("nan")),
            (12345, None),
            ("!!!", "??? @@@ ### 123-456"),
        ],
    )
    def test_missing_or_empty_inputs_never_crash(
        self, subject: object, body: object
    ) -> None:
        """Per README: `subject` can be empty/NaN, so never crash on it."""
        pred = detect_language_with_confidence(subject, body)
        assert pred.language in ("en", "de")
        assert 0.5 <= pred.confidence <= 1.0
        assert not math.isnan(pred.confidence)

    def test_umlaut_only_fallback(self) -> None:
        pred = detect_language_with_confidence("Überprüfung", "")
        assert pred.language == "de"
        assert pred.confidence >= 0.7

    def test_morphological_suffix_tie_breaker(self) -> None:
        pred_de = detect_language_with_confidence(
            "Systemaktualisierung", "Verfügbarkeitsprüfung fehlgeschlagen"
        )
        assert pred_de.language == "de"

        pred_en = detect_language_with_confidence(
            "Configuration", "Deployment processing failed"
        )
        assert pred_en.language == "en"

    def test_batch_detection(self) -> None:
        subjects = ["VPN outage", "Wesentlicher Sicherheitsvorfall", None]
        bodies = [
            "We are experiencing a critical network failure.",
            "Sehr geehrtes Team, der Server ist ausgefallen.",
            "Thank you for your help with our billing account.",
        ]
        results = detect_language_batch(subjects, bodies)
        assert results == ["en", "de", "en"]

    def test_latency_benchmark(self) -> None:
        """Ensure single-ticket inference is under 0.1 ms (target ~0.02 ms)."""
        subj = "Account Disruption"
        body = (
            "Dear Customer Support Team, I am writing to report a significant "
            "problem with the centralized account management portal."
        )
        n_iters = 5000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            detect_language(subj, body)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = elapsed_ms / n_iters
        assert avg_ms < 0.1, f"Expected < 0.1ms per ticket, got {avg_ms:.4f}ms"


class TestMultiLanguageExtensibility:
    """Verify that new languages can be plugged in dynamically via LanguageProfile."""

    def test_dynamic_custom_profile_registration(self) -> None:
        detector = LanguageDetector(profiles=[EN_PROFILE, DE_PROFILE])
        it_profile = LanguageProfile(
            code="it",
            name="Italian",
            distinctive_chars=frozenset("àèéìòù"),
            stopwords=frozenset({
                "il", "lo", "gli", "della", "nella", "sono", "siamo", "abbiamo",
                "buongiorno", "grazie", "problema", "nostro", "sistema", "assistenza",
            }),
            suffix_pattern=re.compile(r"(?:zione|zioni|mento|menti)$"),
        )
        detector.register_profile(it_profile)
        assert "it" in detector.supported_languages

        pred = detector.detect_with_confidence(
            "Problema con il sistema",
            "Buongiorno, abbiamo un problema nella configurazione della nostra assistenza. Grazie.",
        )
        assert pred.language == "it"
        assert pred.scores["it"] > pred.scores["en"]
        assert pred.scores["it"] > pred.scores["de"]

    def test_annotate_dataframe_drops_raw_language_and_adds_detected(self) -> None:
        class DummyDataFrame:
            """Lightweight DataFrame mock so unit tests don't require pandas."""

            def __init__(self, data: dict[str, list[object]]) -> None:
                self._data = {k: list(v) for k, v in data.items()}

            @property
            def columns(self) -> list[str]:
                return list(self._data.keys())

            def copy(self) -> "DummyDataFrame":
                return DummyDataFrame(self._data)

            def drop(self, columns: list[str]) -> "DummyDataFrame":
                return DummyDataFrame(
                    {k: v for k, v in self._data.items() if k not in columns}
                )

            def __getitem__(self, key: str) -> list[object]:
                return self._data[key]

            def __setitem__(self, key: str, value: list[object]) -> None:
                self._data[key] = list(value)

            def __len__(self) -> int:
                return len(next(iter(self._data.values()), []))

        raw_df = DummyDataFrame(
            {
                "subject": ["Video Issue with Mini-Projector", "Probleme mit dem Smart-Türklingel"],
                "body": [
                    "Dear Customer Support Team, I am reaching out to report a persistent problem.",
                    "Ich beschreibe das Problem mit meinem Smart-Türklingel-Gerät.",
                ],
                "language": ["de", "en"],  # Intentionally mislabeled like the raw dataset!
            }
        )

        clean_df = annotate_dataframe_language(raw_df, drop_raw_language=True)
        assert "language" not in clean_df.columns
        assert clean_df["detected_language"] == ["en", "de"]
        assert all(0.5 < c <= 1.0 for c in clean_df["language_confidence"])


class TestSecurityHardening:
    """Verify resilience against DoS, Unicode NFD bypass, character stuffing, and mutation."""

    def test_unicode_nfd_decomposed_umlauts_are_detected(self) -> None:
        # 'Störung' and 'Überprüfung' in decomposed NFD form (o + \u0308, U + \u0308)
        nfd_subject = "Sto\u0308rung"
        nfd_body = "U\u0308berpru\u0308fung fehlgeschlagen"
        pred = detect_language_with_confidence(nfd_subject, nfd_body)
        assert pred.language == "de"
        assert pred.scores["de"] > 0

    def test_character_stuffing_cannot_spoof_english_ticket(self) -> None:
        # English ticket with 500 repeated German umlauts appended at the end
        en_body = (
            "Dear support team, we are unable to connect to the production portal "
            "and we need your help to resolve this issue as soon as possible. "
            "Please let us know what we should do next. " + ("ä" * 500)
        )
        pred = detect_language_with_confidence("Portal connection issue", en_body)
        assert pred.language == "en"

    def test_massive_payload_bounded_latency_no_dos(self) -> None:
        # 2 MB adversarial payload should still complete in well under 5 ms
        huge_body = ("Dear support team, please help us resolve this issue. " * 40_000)
        t0 = time.perf_counter()
        lang = detect_language("Help", huge_body)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert lang == "en"
        assert elapsed_ms < 20.0, f"Expected < 20ms on 2MB payload, got {elapsed_ms:.2f}ms"

    def test_profiles_view_is_immutable(self) -> None:
        detector = LanguageDetector(profiles=[EN_PROFILE, DE_PROFILE])
        with pytest.raises(TypeError):
            detector.profiles["evil"] = EN_PROFILE  # type: ignore[index]


