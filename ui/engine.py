"""Pipeline behind the console:
mask PII -> guardrails -> language -> triage -> search -> burst -> gate -> draft -> verify.

Team modules are used wherever they exist: pii.mask, sentineldesk/language.py, sentineldesk/triage (Triage),
retrieve.search (when pushed). Everything else is a stand-in in ui/ (search.py, gate.py, drafting.py), and
`Desk.sources` says which is which; the console shows it.

    desk = Desk.load()                                  # Streamlit: wrap in st.cache_resource
    a = desk.analyse(subject, body)                     # models + search, once per ticket
    gate, draft = desk.decide(a, cfg)                   # cheap; re-run whenever a sidebar threshold moves

SENTINELDESK_UI_MOCK=1 forces mocks for triage and search (UI work and tests without models).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from sentineldesk.language import detect_language_with_confidence
from sentineldesk.triage.text import build_text
from ui import mocks
from ui.burst import BurstDetector
from ui.drafting import draft_reply
from ui.gate import GateConfig, GateDecision, apply_post_generation, evaluate_gate
from ui.guardrails import SENSITIVE_REQUESTS, detect_injection, mask_with_counts, match_lexicon

REPO = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO / "artifacts" / "ui" / "gate_report.json"
HOLDOUT_METRICS = REPO / "results" / "triage_metrics.json"
TASKS = ("type", "queue", "priority")


def _bow_encode(texts: list[str]) -> np.ndarray:
    """Mock-mode stand-in for the embedder (hashed bag of words), so burst detection works without models."""
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w{3,}", t.lower()):
            out[i, zlib.crc32(w.encode()) % 512] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class Desk:
    def __init__(self, triage=None, search=None, encode=None, sources=None, report=None, cfg=None,
                 burst_threshold: float | None = None):
        self.triage = triage
        self._search = search or mocks.mock_search
        self.encode = encode or _bow_encode
        self.sources = sources or {}
        self.report = report or {}
        self.cfg = cfg or GateConfig()
        thr = burst_threshold or self.report.get("burst", {}).get("threshold") or 0.92
        self.burst = BurstDetector(sim_threshold=thr)
        self._counter = itertools.count(1)
        self._drafts: dict = {}
        self.llm = self.check = None  # set by load(): sentineldesk.draft.draft + sentineldesk.verify.verify

    @classmethod
    def load(cls, artifacts: str = "artifacts", index_dir: str = "data/index") -> "Desk":
        force_mock = os.environ.get("SENTINELDESK_UI_MOCK") == "1"
        src = {"PII mask": "real (pii.py)", "Language": "real (language.py)"}
        report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() and not force_mock else {}
        cfg = GateConfig()
        triage = encode = search = None

        if not force_mock and (REPO / artifacts / "triage" / "meta.json").exists():
            from sentineldesk.triage.predict import Triage

            triage = Triage.load(REPO / artifacts)
            cfg.critical_threshold = float(triage.thresholds["critical"]["threshold"])
            cfg.min_type_confidence = float(triage.thresholds.get(triage.mode, {}).get("type", cfg.min_type_confidence))
            cfg.min_queue_confidence = float(report.get("floors", {}).get("queue", {}).get("chosen", cfg.min_queue_confidence))
            encode = triage.embedder.encode
            src["Triage"] = f"real ({triage.mode} ensemble)" if triage.mode == "full" else f"real ({triage.mode})"
        else:
            src["Triage"] = "mock"

        if not force_mock:
            try:
                import retrieve  # locked interface (Chirag)

                retriever = retrieve.Retriever(embedder=triage.embedder if triage else None)  # share one e5 model
                search = _with_ticket_details(retriever)
                encode = encode or retriever.embedder.encode
                cfg.min_similarity = float(retrieve.SIGMA)
                src["Search"] = "real (hybrid, retrieve.py)"
            except ImportError:
                idx = REPO / index_dir
                if (idx / "emb.npy").exists():
                    from ui.search import DenseSearch

                    meta = json.loads((idx / "meta.json").read_text())
                    if triage is None or (triage.meta.get("embedding_model"), triage.meta.get("prefix")) != \
                            (meta["embedding_model"], meta["prefix"]):
                        from sentineldesk.triage.embed import Embedder

                        encode = Embedder(meta["embedding_model"], meta["prefix"]).encode
                    search = DenseSearch(idx, encode).search
                    src["Search"] = "stand-in (dense only)"
        if search is None:
            src["Search"] = "mock"

        src["Gate"] = "real (ui/gate.py)"
        llm = check = None
        if not force_mock and os.environ.get("SENTINELDESK_LLM", "1") != "0" and os.environ.get("GEMINI_API_KEY"):
            try:
                from sentineldesk.draft import draft as llm
                from sentineldesk.verify import verify as check
            except ImportError:
                llm = check = None
        src["Draft + verify"] = ("real (draft.py Gemini + verify.py; offline draft if the LLM fails)" if llm
                                 else "offline (ui/drafting.py; set GEMINI_API_KEY, SENTINELDESK_LLM!=0)")
        src["Burst detection"] = "bonus (ui/burst.py)" if encode else "bonus (mock vectors)"
        desk = cls(triage, search, encode, src, report, cfg)
        desk.llm, desk.check = llm, check
        if encode:
            encode(["warm-up"])  # load the encoder once so the first live ticket isn't slow
        return desk

    # ---- slow part: once per ticket ---------------------------------------------------------
    def analyse(self, subject: str | None, body: str | None, *, k: int = 5, cross_lingual: bool = True,
                track_burst: bool = True, exclude_ids=None, on_stage=None) -> dict:
        """on_stage(key, a) is called as each stage finishes (live pipeline strip); `a` fills up as it goes."""
        subject, body = (subject or "").strip(), (body or "").strip()
        if not (subject or body):
            raise ValueError("Enter a subject or a body.")
        emit = on_stage or (lambda *_: None)
        t: dict[str, float] = {}
        t0 = s = time.perf_counter()
        a = {"ref": f"LIVE-{next(self._counter):04d}", "subject": subject, "body": body, "timings_ms": t}

        m_subject, c1 = mask_with_counts(subject)
        m_body, c2 = mask_with_counts(body)
        raw, masked = build_text(subject, body), build_text(m_subject, m_body)
        a.update(masked_text=masked, pii_counts=dict(Counter(c1) + Counter(c2)))
        t["mask"] = _ms(s)
        emit("mask", a)

        s = time.perf_counter()
        lp = detect_language_with_confidence(m_subject, m_body)
        a.update(language=lp.language, language_confidence=round(float(lp.confidence), 3))
        t["language"] = _ms(s)
        emit("language", a)

        s = time.perf_counter()
        a.update(sensitive_hits=match_lexicon(raw, SENSITIVE_REQUESTS), injection_hits=detect_injection(raw))
        t["guardrails"] = _ms(s)
        emit("guardrails", a)

        s = time.perf_counter()
        tri = self.triage.predict(m_subject, m_body) if self.triage else mocks.mock_triage(m_subject, m_body)
        a["triage"] = tri
        t["triage"] = _ms(s)
        emit("triage", a)

        s = time.perf_counter()
        preds = {h: tri[h]["probs"] for h in ("type", "queue")}
        found = self._search(masked, k=k if cross_lingual else 4 * k, preds=preds, lang=lp.language,
                             exclude_ids=exclude_ids)
        a["precedents"] = found if cross_lingual else [p for p in found if p["language"] == lp.language][:k]
        t["retrieve"] = _ms(s)
        emit("retrieve", a)

        a["burst"] = None
        if track_burst:
            s = time.perf_counter()
            vec = self.encode([masked])[0]
            failure_like = tri["type"]["label"] in ("Incident", "Problem") or tri["critical"]["fires"]
            a["burst"] = self.burst.add(a["ref"], vec, subject, body[:300], failure_like)
            t["burst"] = _ms(s)
        t["total"] = _ms(t0)
        return a

    # ---- cheap part: gate + draft + verify ----------------------------------------------------
    def decide(self, a: dict, cfg: GateConfig | None = None, on_stage=None) -> tuple[GateDecision, object]:
        """on_stage(key, a, gate, draft) for gate / draft / verify / decision (live pipeline strip)."""
        emit = on_stage or (lambda *_: None)
        cfg = cfg or self.cfg
        s = time.perf_counter()
        gate = evaluate_gate(a["triage"], a["precedents"], a["sensitive_hits"], a["injection_hits"], cfg)
        a["timings_ms"]["gate"] = _ms(s)
        a["gate_pre"] = gate.status.value  # before draft checks can raise it (pipeline strip "Gate" box)
        emit("gate", a, gate, None)
        key = (a["ref"], gate.status, gate.has_precedent)
        draft = self._drafts.get(key)
        if draft is None:  # cached per situation, so moving a slider never re-drafts the same ticket
            s = time.perf_counter()
            draft = draft_reply(a["masked_text"], a["language"], a["precedents"], gate.status.value,
                                gate.has_precedent, bool(a["injection_hits"]), self.llm, self.check)
            total = _ms(s)
            draft.timings.setdefault("draft", total - draft.timings.get("verify", 0.0))
            draft.timings.setdefault("verify", max(total - draft.timings["draft"], 0.0))
            self._drafts[key] = draft
        a["timings_ms"]["draft"], a["timings_ms"]["verify"] = draft.timings["draft"], draft.timings["verify"]
        emit("draft", a, gate, draft)
        emit("verify", a, gate, draft)
        apply_post_generation(gate, draft.issues, draft.unsupported_requests)
        emit("decision", a, gate, draft)
        return gate, draft

    def process(self, subject, body, cfg=None, **kw):
        a = self.analyse(subject, body, **kw)
        return a, *self.decide(a, cfg)

    # ---- sidebar metrics ------------------------------------------------------------------------
    def metrics(self) -> dict:
        """Holdout metrics once scripts/eval_triage.py has run (once, at the end); half-B metrics before that."""
        if HOLDOUT_METRICS.exists():
            m = json.loads(HOLDOUT_METRICS.read_text())
            # eval_triage.py labels variants e.g. "full ensemble (0+1+2+6)"; show the one the live console runs
            variant = next(v for v in m["macro_f1"]["type"] if v.startswith(m["mode"]))
            return {"title": "Holdout metrics", "n": m.get("n_test"),
                    "macro_f1": {h: m["macro_f1"][h].get(variant) for h in TASKS},
                    "crit_recall": m["high_priority_incident"]["critical_detector"]["recall"]}
        if self.triage is None:
            return {}
        calib = self.triage.calib.get(self.triage.mode, {})
        return {"title": "Validation metrics (half B)", "n": None,
                "macro_f1": {h: calib.get(h, {}).get("macro_f1") for h in TASKS},
                "crit_recall": self.triage.thresholds["critical"].get("half_b_recall")}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _with_ticket_details(retriever):
    """retrieve.search() returns the locked fields; the console's "Full ticket" view also shows the customer's
    message and tags, looked up in the same index rows."""
    kb, row = retriever.kb, retriever.row
    tags = [f"tag_{i}" for i in range(1, 9)]

    def search(text, k=5, preds=None, lang=None, exclude_ids=None, pool_ids=None):
        hits = retriever.search(text, k, preds, lang, exclude_ids, pool_ids)
        for h in hits:
            r = kb.iloc[row[h["ticket_id"]]]
            full, subj = r["text"], h.get("subject") or ""
            h["body"] = full[len(subj) + 1:] if full.startswith(subj + "\n") else full
            h["tags"] = [t for t in (r[c] for c in tags) if isinstance(t, str) and t]
        return hits

    return search
