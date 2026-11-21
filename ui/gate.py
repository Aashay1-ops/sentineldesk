"""Escalation gate (stand-in until the team's gate module exists; README status board "Escalation gate").

Three outcomes, highest severity wins:
    ESCALATE_ON_CALL  - page the on-call engineer now (the draft is an acknowledgement only)
    HUMAN_REVIEW      - an agent must review / complete the reply before it is sent
    AUTO_DRAFT_READY  - grounded draft can be sent after a quick skim

Rules (each produces a reason code + evidence for the console):
  CRITICAL_SCORE     triage's critical detector score >= threshold (recall-tuned on half B). Per the team
                     decision the detector replaces keyword overrides, so outage keywords are evidence only.
  LOW_CONFIDENCE     type or queue confidence below its floor
  NO_PRECEDENT       best past ticket's dense cosine < sigma -> nothing to ground a reply on
  SENSITIVE_REQUEST  refund / pricing / legal-SLA / data-deletion ask: only a person may grant it, whatever a
                     past answer said (a past refund was for a different case)
  PROMPT_INJECTION   instructions aimed at the model inside the ticket; also demotes any on-call reason,
                     so a crafted ticket cannot page the on-call engineer
  DRAFT_GUARD        (after drafting) the draft failed verification
  LLM_FLAGGED        (after drafting) the model reported asks it could not ground
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    AUTO = "AUTO_DRAFT_READY"
    REVIEW = "HUMAN_REVIEW"
    ONCALL = "ESCALATE_ON_CALL"


_RANK = {Status.AUTO: 0, Status.REVIEW: 1, Status.ONCALL: 2}


@dataclass
class Reason:
    code: str
    severity: Status
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class GateDecision:
    status: Status = Status.AUTO
    reasons: list[Reason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    has_precedent: bool = True

    def add(self, reason: Reason) -> None:
        self.reasons.append(reason)
        if _RANK[reason.severity] > _RANK[self.status]:
            self.status = reason.severity


@dataclass
class GateConfig:
    critical_threshold: float = 0.44     # triage: catches >= 90% of high-priority Incidents on half B
    min_type_confidence: float = 0.50    # triage's suggested threshold (half B)
    min_queue_confidence: float = 0.30   # scripts/build_ui_report.py (half B)
    min_similarity: float = 0.87         # sigma, README "No precedent threshold" for e5-base
    top_k: int = 5


def evaluate_gate(tri: dict, precedents: list[dict], sensitive_hits: dict[str, list[str]],
                  injection_hits: list[str], cfg: GateConfig) -> GateDecision:
    d = GateDecision()
    t, q = tri["type"], tri["queue"]
    c = tri["critical"]

    if c["score"] >= cfg.critical_threshold:
        kw = c.get("keyword_hits") or []
        d.add(Reason("CRITICAL_SCORE", Status.ONCALL,
                     f"Critical-incident score {c['score']:.0%} ≥ threshold {cfg.critical_threshold:.0%}"
                     + (f" · outage keywords: {', '.join(kw)}" if kw else ""),
                     {"critical_score": c["score"], "keyword_hits": kw}))

    low = {}
    if t["confidence"] < cfg.min_type_confidence:
        low["type"] = t["confidence"]
    if q["confidence"] < cfg.min_queue_confidence:
        low["queue"] = q["confidence"]
    if low:
        d.add(Reason("LOW_CONFIDENCE", Status.REVIEW,
                     "Low triage confidence on " + ", ".join(f"{k} ({v:.0%})" for k, v in low.items()), low))

    best = precedents[0]["similarity"] if precedents else 0.0
    if best < cfg.min_similarity:
        d.has_precedent = False
        d.add(Reason("NO_PRECEDENT", Status.REVIEW,
                     f"No sufficiently similar resolved ticket (best {best:.3f} < σ {cfg.min_similarity:.2f}); "
                     "likely out of scope - holding reply only", {"best_similarity": best}))

    if sensitive_hits:
        cats = ", ".join(cat.replace("_", "/") for cat in sensitive_hits)
        d.add(Reason("SENSITIVE_REQUEST", Status.REVIEW,
                     f"Customer asks for {cats} - only a person may approve this; the draft must not commit",
                     sensitive_hits))
        for cat in sensitive_hits:
            from ui.guardrails import SENSITIVE_REQUESTS  # local import keeps this module dependency-free

            ids = [p["ticket_id"] for p in precedents if SENSITIVE_REQUESTS[cat].search(p.get("answer") or "")]
            if ids:
                d.notes.append(f"{cat.replace('_', '/')} is handled in precedent(s) {', '.join(ids)}: the agent can "
                               "follow that process, but amounts and guarantees need approval.")

    if injection_hits:
        demoted = [r for r in d.reasons if r.severity == Status.ONCALL]
        for r in demoted:
            r.severity = Status.REVIEW
            r.message += " (demoted: adversarial input cannot page on-call)"
        if demoted:
            d.status = max((r.severity for r in d.reasons), key=lambda s: _RANK[s])
        d.add(Reason("PROMPT_INJECTION", Status.REVIEW,
                     "Ticket contains instructions aimed at the AI - treated as data, routed to a human",
                     {"matches": injection_hits}))
    return d


def apply_post_generation(d: GateDecision, issues: list[str], unsupported_asks: list[str]) -> None:
    if issues:
        d.add(Reason("DRAFT_GUARD", Status.REVIEW, "Draft failed verification: " + "; ".join(issues), {"issues": issues}))
    if unsupported_asks:
        d.add(Reason("LLM_FLAGGED", Status.REVIEW, "Model could not ground: " + "; ".join(unsupported_asks),
                     {"unsupported_asks": unsupported_asks}))
