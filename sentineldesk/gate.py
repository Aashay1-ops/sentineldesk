"""Gate as a YES/NO checklist: "May the AI write the reply?"

Every question is phrased so that YES = safe. All answers must be YES for the AI to draft (and, after drafting,
for the draft to be shown). The first NO becomes the escalation reason; every NO is listed with its evidence.

Before drafting (gate)                                   code if NO
  Q1  Is there a readable ticket?                         INPUT
  Q2  Is it in English or German?                         INPUT
  Q3  Is it free of instructions aimed at the AI?         INPUT
  Q4  Is it free of signs of a critical outage?           CRITICAL
  Q5  Is the triage confident?                            LOW_CONFIDENCE
  Q6  Is there a close past ticket?                       NO_PRECEDENT
  Q7  Does it ask only for things past tickets did?       UNSUPPORTED_ASK
After drafting (verify)
  Q8  Does every citation point to a retrieved ticket?    VERIFY_FAILED
  Q9  Do the past tickets cover the request?              NO_PRECEDENT
  Q10 Did the draft avoid unsupported promises?           UNSUPPORTED_ASK
  Q11 Did the draft come back clean (no error/injection)? VERIFY_FAILED
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

AUTO = "AUTO_DRAFT_READY"
REVIEW = "HUMAN_REVIEW"
ONCALL = "ESCALATE_ON_CALL"
_RANK = {AUTO: 0, REVIEW: 1, ONCALL: 2}

INJECTION_RE = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|disregard (the |your )?(rules|instructions)|"
    r"system prompt|you are now|act as|jailbreak|ignoriere (alle )?(vorherigen )?anweisungen|systemprompt",
    re.I,
)
# Outage words. NOTE: the team repo replaces this with its trained critical detector (critical.fires).
CRITICAL_RE = re.compile(
    r"\b(outage|is down|went down|are down|offline|not reachable|unreachable|all (users|staff|employees)|"
    r"company-?wide|entire (team|site|office)|production (is )?down|data (loss|breach)|breach|ransomware|"
    r"security incident|patient|hospital|clinic|emr|ehr|life-critical|cannot (process|access) (any|payments)|"
    r"ausfall|ausgefallen|nicht erreichbar|alle (mitarbeiter|benutzer|nutzer)|produktion steht|datenverlust|"
    r"sicherheitsvorfall|patienten|krankenhaus|klinik)\b",
    re.I,
)
# Requests that need a commitment only a human / policy owner can make.
POLICY_RE = re.compile(
    r"refund|money back|compensat|reimburs|discount|price|pricing|quote|invoice amount|sla credit|guarantee|"
    r"lawsuit|lawyer|legal action|attorney|gdpr|delete (all )?my data|cancel (my )?contract|"
    r"rückerstattung|erstattung|entschädigung|rabatt|preis|angebot|gutschrift|anwalt|klage|rechtlich|dsgvo|kündig",
    re.I,
)


@dataclass
class GateDecision:
    status: str = AUTO
    checks: list[dict] = field(default_factory=list)   # the YES/NO checklist, in order
    reasons: list[str] = field(default_factory=list)   # one per NO
    codes: list[str] = field(default_factory=list)     # one per NO
    flags: dict = field(default_factory=dict)

    def ask(self, qid: str, question: str, yes: bool, evidence: str, code: str, on_no: str = REVIEW,
            stage: str = "gate") -> bool:
        self.checks.append({"id": qid, "stage": stage, "question": question,
                            "answer": "YES" if yes else "NO", "evidence": evidence, "code_if_no": code})
        if not yes:
            if _RANK[on_no] > _RANK[self.status]:
                self.status = on_no
            self.reasons.append(f"{question} NO: {evidence}")
            self.codes.append(code)
        return yes

    def raise_to(self, status: str, reason: str, code: str):
        """For signals outside the checklist (e.g. outage bursts)."""
        if _RANK[status] > _RANK[self.status]:
            self.status = status
        self.reasons.append(reason)
        self.codes.append(code)

    @property
    def ai_may_respond(self) -> bool:
        return not self.codes

    def flag(self) -> dict:
        """The one-line answer the RAG system exposes, plus the checklist behind it."""
        first_no = next((c for c in self.checks if c["answer"] == "NO"), None)
        return {"ai_may_respond": "YES" if self.ai_may_respond else "NO",
                "first_no": first_no["question"] if first_no else (self.reasons[0] if self.reasons else None),
                "checks": self.checks}


def pre_generation(text: str, lang: str, lang_conf: float, triage: dict, top_sim: float) -> GateDecision:
    g = GateDecision()
    G = config.GATE
    n = len(text.strip())

    if not g.ask("Q1", "Is there a readable ticket?", n >= G["min_body_chars"],
                 f"{n} characters (need {G['min_body_chars']})", "INPUT"):
        return g  # nothing else can be judged on an empty ticket; model outputs here are noise
    g.ask("Q2", "Is it in English or German?", lang in config.SUPPORTED_LANGS,
          f"detected '{lang}' (p={lang_conf:.2f})", "INPUT")
    inj = INJECTION_RE.search(text)
    g.ask("Q3", "Is it free of instructions aimed at the AI?", not inj,
          f"found '{inj.group(0)}'" if inj else "none found", "INPUT")

    p_high = triage["priority"]["probs"].get("high", 0.0)
    high_incident = triage["type"]["label"] == "Incident" and (
        triage["priority"]["label"] == "high" or p_high >= G["high_priority_prob"])
    crit = CRITICAL_RE.findall(text) if triage["type"]["label"] in ("Incident", "Problem") else []
    terms = sorted({(m[0] if isinstance(m, tuple) else m).lower() for m in crit})[:4]
    if terms:
        g.flags["critical_terms"] = terms
    ev = (f"P(high)={p_high:.2f} on an Incident" if high_incident else f"P(high)={p_high:.2f}") + \
         (f"; outage words: {', '.join(terms)}" if terms else "")
    g.ask("Q4", "Is it free of signs of a critical outage?", not (high_incident or terms), ev, "CRITICAL", on_no=ONCALL)

    tc, qc = triage["type"]["confidence"], triage["queue"]["confidence"]
    g.ask("Q5", "Is the triage confident?", tc >= G["min_type_conf"] and qc >= G["min_queue_conf"],
          f"type {tc:.2f} (need {G['min_type_conf']}), queue {qc:.2f} (need {G['min_queue_conf']})", "LOW_CONFIDENCE")
    g.ask("Q6", "Is there a close past ticket?", top_sim >= G["min_top_similarity"],
          f"best similarity {top_sim:.2f} (need {G['min_top_similarity']})", "NO_PRECEDENT")

    pol = POLICY_RE.search(text)
    if pol:
        g.flags["policy_sensitive"] = True
    g.ask("Q7", "Does it ask only for things past tickets did?", not pol,
          f"asks about '{pol.group(0)}' (refund, price, SLA or legal)" if pol else "no refund, price, SLA or legal request",
          "UNSUPPORTED_ASK")
    return g


def post_generation(g: GateDecision, draft) -> GateDecision:
    if draft is None:
        return g
    v = "verify"
    g.ask("Q8", "Does every citation point to a retrieved ticket?", not draft.invalid_citations and bool(draft.cited_ids),
          f"cited {draft.cited_ids or 'nothing'}" + (f"; removed {draft.invalid_citations}" if draft.invalid_citations else ""),
          "VERIFY_FAILED", stage=v)
    g.ask("Q9", "Do the past tickets cover the request?", draft.evidence_sufficient,
          "model says yes" if draft.evidence_sufficient else "model says the precedents don't cover it",
          "NO_PRECEDENT", stage=v)
    g.ask("Q10", "Did the draft avoid unsupported promises?", not draft.unsupported_requests,
          "; ".join(draft.unsupported_requests[:3]) or "none", "UNSUPPORTED_ASK", stage=v)
    clean = not draft.injection_detected and not (draft.error and draft.mode == "llm")
    if draft.injection_detected:
        ev = "model flagged embedded instructions"
    elif draft.mode != "llm":
        ev = f"extractive fallback: closest past answer quoted verbatim ({draft.error})"
    else:
        ev = draft.error or "LLM draft returned valid JSON"
    g.ask("Q11", "Did the draft come back clean?", clean, ev, "VERIFY_FAILED", stage=v)
    return g


_VERIFY_QUESTIONS = {
    "format": "Is the draft well-formed?",
    "citations": "Is every cited ticket one that was retrieved?",
    "numbers": "Is every number, amount, date and duration grounded in a cited answer?",
    "placeholders": "Is the draft free of invented personal data and links?",
    "language": "Is the draft in the customer's language?",
    "unsupported_asks": "Did the draft avoid promises no past ticket supports?",
}


def apply_verify(g: GateDecision, result: dict) -> GateDecision:
    """Add verify()'s six checks to the YES/NO checklist; each failure escalates with verify's own code."""
    codes = {f["check"]: f["code"] for f in result["failures"]}
    for i, c in enumerate(result["checks"], 1):
        g.ask(f"V{i}", _VERIFY_QUESTIONS.get(c["check"], c["check"]), c["passed"], c["evidence"],
              codes.get(c["check"], "VERIFY_FAILED"), stage="verify")
    return g
