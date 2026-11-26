"""End-to-end: detect language -> triage -> retrieve -> gate -> grounded draft -> gate."""
from __future__ import annotations

import time
from dataclasses import asdict

from . import escalation as escalate
from . import gate as gate_mod
from .burst import BurstDetector
from .language import detect_language_with_confidence
from .pii import mask_pii
from .data import ticket_text
from .generator import draft_reply
from .retrieval import Retriever
from .triage import TriageModel
from .verify import verify

def detect_language(text: str) -> tuple[str, float]:
    p = detect_language_with_confidence(body=text)
    return p.language, p.confidence


class SentinelDesk:
    def __init__(self, escalation_mode: str | None = None, log_escalations: bool = True):
        self.escalation_mode = escalation_mode or escalate.MODE  # "severity" or "simple"
        self.log_escalations = log_escalations
        self.triage = TriageModel.load()
        self.retriever = Retriever()
        self.burst = BurstDetector()
        self.retriever.search("warm-up", "load the encoder before the first real ticket")

    def process(self, subject: str | None, body: str | None, generate: bool = True) -> dict:
        subject, body = (subject or "").strip(), (body or "").strip()
        text = ticket_text(subject, body)
        timings = {}
        t0 = time.perf_counter()

        lang, lang_conf = detect_language(text) if text else ("unknown", 0.0)
        timings["lang_ms"] = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        hits, qvec = self.retriever.search(subject, body) if text else ([], None)
        assert all(h["ticket_id"] in self.retriever.kb_ids for h in hits)  # leakage guard: KB only
        timings["retrieval_ms"] = (time.perf_counter() - t) * 1000
        # RRF only orders hits; the "no precedent" check uses the best dense cosine among them.
        top_sim = max((h["similarity"] for h in hits), default=0.0)

        t = time.perf_counter()
        # Label votes use dense neighbours (what the blend was tuned on); hybrid hits feed the draft.
        voters = self.retriever.search_with_vec(text, qvec, mode="dense") if qvec is not None else []
        tri = self.triage.predict_one(subject, body, precedents=voters)
        top_sim = max([top_sim] + [v["similarity"] for v in voters[:1]])  # closest precedent overall
        timings["triage_ms"] = (time.perf_counter() - t) * 1000

        g = gate_mod.pre_generation(text, lang, lang_conf, tri, top_sim)

        burst = None
        if qvec is not None and tri["type"]["label"] in ("Incident", "Problem"):
            burst = self.burst.observe(qvec, subject or body[:60])
            if burst:
                g.raise_to(gate_mod.ONCALL, f"Major incident: {burst['count']} similar tickets in {burst['window_min']} min", "BURST")

        # README: a ticket the gate escalates never reaches the LLM. Only clean tickets are drafted,
        # and verify (post_generation) can still escalate them.
        draft = None
        if generate and hits and not g.codes:
            t = time.perf_counter()
            draft = draft_reply(subject, body, lang, {k: v["label"] for k, v in tri.items()}, hits)
            timings["llm_ms"] = (time.perf_counter() - t) * 1000
            g = gate_mod.post_generation(g, draft)
        draft_d = asdict(draft) if draft else None

        masked, _ = mask_pii(text)
        verification = None
        if draft is not None and draft.reply:
            # ⑤ VERIFY: deterministic rules on the draft; every failed check escalates with its reason
            t = time.perf_counter()
            verification = verify({"reply": draft.reply, "cited_ids": draft.cited_ids,
                                   "unsupported_asks": draft.unsupported_requests}, hits, masked, lang)
            timings["verify_ms"] = (time.perf_counter() - t) * 1000
            g = gate_mod.apply_verify(g, verification)
        flag = g.flag()  # {"ai_may_respond": "YES"/"NO", "first_no": ..., "checks": [...]}
        escalation = escalate.route(g.codes, g.reasons, tri, hits, lang, masked, draft=draft_d,
                                    mode=self.escalation_mode, log=self.log_escalations, checklist=flag["checks"])

        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {
            "language": lang,
            "language_conf": lang_conf,
            "triage": tri,
            "gate": {"status": g.status, "reasons": g.reasons, "codes": g.codes, "flags": g.flags},
            "outcome": "ESCALATE" if escalation else "AUTO_DRAFT_READY",
            "ai_may_respond": flag["ai_may_respond"],
            "first_no": flag["first_no"],
            "checklist": flag["checks"],
            "escalation": escalation,
            "precedents": hits,
            # An escalated ticket's draft is never shown as a reply; a verify-blocked one lives in the record.
            "draft": None if escalation else draft_d,
            "verification": verification,
            "burst": burst,
            "timings": timings,
        }
