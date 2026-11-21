"""Offline tests for sentineldesk/draft.py (no network, no API key)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk import draft as d  # noqa: E402

HITS = [{"ticket_id": "TKT-00001", "language": "en", "subject": "VPN", "answer": "Dear [Name], restart the VPN client."},
        {"ticket_id": "TKT-00002", "language": "de", "subject": "VPN", "answer": float("nan")}]


def test_prompt_has_sources_language_and_normalized_placeholders():
    system, user = d.build_prompt("VPN down", HITS, "de")
    assert "German" in system and 'id="TKT-00001"' in user and "<name>" in user and "[Name]" not in user
    assert "TKT-00002" not in user  # empty (NaN) answer is not offered as a source


def test_no_usable_source_raises():
    with pytest.raises(d.DraftError):
        d.build_prompt("x", [{"ticket_id": "TKT-1", "answer": None}], "en")
    with pytest.raises(ValueError):
        d.build_prompt("x", HITS, "fr")


@pytest.mark.parametrize("bad", ["not json", "[]", '{"reply": 1, "cited_ids": [], "unsupported_asks": []}',
                                 '{"reply": "hi", "cited_ids": "TKT-1", "unsupported_asks": []}'])
def test_parse_rejects_bad_shapes(bad):
    with pytest.raises(d.DraftError):
        d.parse(bad)


def test_fallback_model_used_when_primary_fails(monkeypatch):
    calls = []

    def fake_call(model, system, user):
        calls.append(model)
        if model == d.MODEL:
            raise d.DraftError(f"{model}: HTTP 503")   # non-429 errors fall back straight away
        return json.dumps({"reply": "Restart the VPN client [TKT-00001].", "cited_ids": ["TKT-00001"],
                           "unsupported_asks": []})

    monkeypatch.setattr(d, "MODEL", "primary-model")
    monkeypatch.setattr(d, "_call", fake_call)
    out = d.draft("VPN down", HITS, "en")
    assert calls == ["primary-model", d.FALLBACK_MODEL] and out["model"] == d.FALLBACK_MODEL
    assert set(out) == {"reply", "cited_ids", "unsupported_asks", "model", "latency_ms"}


def test_rate_limit_retries_same_model_once(monkeypatch):
    calls = []

    def fake_call(model, system, user):
        calls.append(model)
        if len(calls) == 1:
            raise d.DraftError(f"{model}: HTTP 429")
        return json.dumps({"reply": "ok [TKT-00001]", "cited_ids": ["TKT-00001"], "unsupported_asks": []})

    monkeypatch.setattr(d, "RETRY_429_S", 0)
    monkeypatch.setattr(d, "_call", fake_call)
    assert d.draft("VPN down", HITS, "en")["model"] == d.MODEL and calls == [d.MODEL, d.MODEL]


def test_all_models_failing_raises(monkeypatch):
    monkeypatch.setattr(d, "_call", lambda *a: (_ for _ in ()).throw(d.DraftError("down")))
    with pytest.raises(d.DraftError, match="down"):
        d.draft("VPN down", HITS, "en")


def test_missing_key_is_a_draft_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(d.DraftError, match="GEMINI_API_KEY"):
        d._call("m", "s", "u")
