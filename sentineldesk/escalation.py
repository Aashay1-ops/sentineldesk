"""What happens after "Escalate".

Every escalation, in either mode, gets:
  1. a destination (+ a severity in "severity" mode),
  2. a fixed, templated acknowledgement for the customer (never LLM text, never a promise),
  3. a record appended to data/escalations.jsonl (PII-masked text only) for the console's handoff panel.

Two modes, chosen with SENTINEL_ESCALATION_MODE:
  "severity" : P1 / P2 / P3, each with its own destination (default)
  "simple"   : one outcome, "Escalate to On-Call Engineer", with all reasons listed (the brief's badge)
Reason codes and reason strings are identical in both modes; only routing and the badge differ.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import threading
from datetime import datetime, timezone

from . import config

MODE = os.getenv("SENTINEL_ESCALATION_MODE", "severity")
LOG_PATH = config.DATA_DIR / "escalations.jsonl"

# Reason code -> severity (1 = page now, 2 = policy owner decides, 3 = team queue)
SEVERITY = {
    "CRITICAL": 1,
    "BURST": 1,
    "UNSUPPORTED_ASK": 2,
    "VERIFY_FAILED": 3,
    "NO_PRECEDENT": 3,
    "LOW_CONFIDENCE": 3,
    "INPUT": 3,
}
ONCALL = "On-call engineer"
TRIAGE_DESK = "Human triage desk"

_BILLING_RE = re.compile(r"refund|money back|reimburs|invoice|charge|billing|rückerstattung|erstattung|rechnung|abbuchung|gutschrift", re.I)
_SALES_RE = re.compile(r"price|pricing|quote|discount|offer|feature|preis|angebot|rabatt|funktion", re.I)
_LEGAL_RE = re.compile(r"lawsuit|lawyer|legal|attorney|gdpr|anwalt|klage|rechtlich|dsgvo", re.I)

ACK = {
    "en": ("Thank you for contacting us. Your request has been passed to a specialist on our team, "
           "who will review it personally. Your reference is {ref}."),
    "de": ("Vielen Dank für Ihre Nachricht. Ihr Anliegen wurde an eine Fachkraft in unserem Team weitergeleitet, "
           "die es persönlich prüft. Ihre Referenznummer lautet {ref}."),
}
ACK_URGENT = {"en": "We are treating your request as urgent.",
              "de": "Wir behandeln Ihr Anliegen mit hoher Dringlichkeit."}

_counter = itertools.count(1)
_lock = threading.Lock()


def _policy_owner(text: str) -> str:
    if _LEGAL_RE.search(text):
        return "Customer Service lead (legal)"
    if _BILLING_RE.search(text):
        return "Billing and Payments"
    if _SALES_RE.search(text):
        return "Sales and Pre-Sales"
    return "Customer Service lead"


def _destination(severity: int, codes: list[str], queue: str, text: str) -> str:
    if severity == 1:
        return ONCALL
    if severity == 2:
        return _policy_owner(text)
    # P3: a person must pick the team when triage itself is unreliable
    if any(c in ("LOW_CONFIDENCE", "INPUT") for c in codes):
        return TRIAGE_DESK
    return f"{queue} team"


def acknowledgement(lang: str, ref: str, urgent: bool) -> str:
    langs = [lang] if lang in ACK else ["en", "de"]  # unknown language: send both
    parts = []
    for l in langs:
        msg = ACK[l].format(ref=ref)
        if urgent:
            msg += " " + ACK_URGENT[l]
        parts.append(msg)
    return "\n\n".join(parts)


def route(codes: list[str], reasons: list[str], triage: dict, precedents: list[dict], language: str,
          masked_text: str, draft: dict | None = None, mode: str | None = None, log: bool = True,
          checklist: list[dict] | None = None) -> dict | None:
    """Return the escalation record, or None when the ticket is Auto-Draft Ready."""
    if not codes:
        return None
    mode = mode or MODE
    worst = min(SEVERITY.get(c, 3) for c in codes)
    queue = triage.get("queue", {}).get("label", "Customer Service")

    if mode == "simple":
        severity, route_to = None, ONCALL
        badge = "Escalate to On-Call Engineer"
        urgent = worst == 1
    else:
        severity, route_to = f"P{worst}", _destination(worst, codes, queue, masked_text)
        badge = "Escalate to On-Call Engineer" if worst == 1 else f"Escalate to {route_to}"
        urgent = worst == 1

    ref = f"ESC-{next(_counter):04d}"
    record = {
        "escalation_id": ref,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "severity": severity,
        "route_to": route_to,
        "badge": badge,
        "ai_may_respond": "NO",
        "reasons": [{"code": c, "text": r} for c, r in zip(codes, reasons)],
        "checklist": checklist or [],
        "language": language,
        "ticket_masked": masked_text,
        "triage": {k: {"label": v["label"], "confidence": round(v["confidence"], 3)} for k, v in triage.items()},
        "precedents": [{"ticket_id": p["ticket_id"], "similarity": round(p["similarity"], 3), "queue": p["queue"],
                        "answer_snippet": str(p["answer"])[:200]} for p in precedents],
        # Only a draft blocked by verify is shown to the agent; gate escalations never get one.
        "draft": draft if (draft and "VERIFY_FAILED" in codes) else None,
        "acknowledgement": acknowledgement(language, ref, urgent),
        "status": "open",
        "agent_action": None,
    }
    if log:
        with _lock:
            config.DATA_DIR.mkdir(exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
