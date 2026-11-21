"""Fake triage and search results with the exact shapes of the real interfaces.

Used when the real piece is not available (no trained artifacts, no index) or when
SENTINELDESK_UI_MOCK=1 is set. Shapes:
  - mock_triage   -> Triage.predict()  (sentineldesk/triage/README.md)
  - mock_search   -> search()           (README "Locked interface: search()")
Values are keyword-driven so the demo paths behave plausibly; they are not model output.
"""
from __future__ import annotations

import re

TYPES = ["Incident", "Request", "Problem", "Change"]
QUEUES = [
    "Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
    "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]
PRIORITIES = ["high", "medium", "low"]

_OUTAGE = re.compile(r"(?i)\b(down|outage|ausfall|cannot connect|can't connect|unreachable|nicht erreichbar|vpn)\b")
_BILLING = re.compile(r"(?i)(invoice|rechnung|charge|abgebucht|refund|erstatt|payment|zahlung)")
_PRODUCT = re.compile(r"(?i)(export|feature|support .* pdf|dashboard|integrat)")

# Canned past tickets for mock mode (invented, short).
_PAST = [
    ("TKT-90001", "en", "Incident", "Service Outages and Maintenance", "high", "VPN connection failing for all users",
     "We have identified a certificate issue on the VPN gateway and rolled back the change. "
     "Please reconnect; if the error persists, restart the client and send us the log file."),
    ("TKT-90002", "de", "Request", "Billing and Payments", "medium", "Doppelte Abbuchung",
     "Vielen Dank für Ihre Nachricht. Wir haben die doppelte Abbuchung geprüft und veranlassen die "
     "Korrektur. Die Gutschrift erscheint innerhalb von 5 Werktagen auf Ihrem Konto."),
    ("TKT-90003", "en", "Request", "Product Support", "low", "Export dashboard as PDF",
     "Yes, reports can be exported as PDF from the dashboard menu (Share > Export > PDF). "
     "Scheduled monthly exports can be set up under Settings > Reports."),
    ("TKT-90004", "en", "Problem", "Technical Support", "medium", "Sync very slow after update",
     "This is a known issue in version 4.2. Please update to 4.2.1, which restores normal sync speed. "
     "Clearing the local cache afterwards also helps."),
    ("TKT-90005", "de", "Incident", "IT Support", "medium", "Drucker druckt leere Seiten",
     "Bitte installieren Sie den aktuellen Druckertreiber neu und führen Sie die Reinigung des "
     "Druckkopfs aus. Falls das Problem bleibt, senden Sie uns die Modellnummer."),
]


def _dist(labels, top, conf):
    rest = (1.0 - conf) / (len(labels) - 1)
    return {l: (conf if l == top else rest) for l in labels}


def _task(labels, top, conf):
    return {"label": top, "confidence": round(conf, 4), "probs": _dist(labels, top, conf)}


def mock_triage(subject: str | None, body: str) -> dict:
    text = f"{subject or ''}\n{body or ''}"
    if not text.strip():
        raise ValueError("empty subject and body")
    outage, billing, product = bool(_OUTAGE.search(text)), bool(_BILLING.search(text)), bool(_PRODUCT.search(text))
    if outage:
        t, q, p, crit, hits = ("Incident", 0.86), ("Service Outages and Maintenance", 0.58), ("high", 0.81), 0.93, ["vpn", "down"]
    elif billing:
        t, q, p, crit, hits = ("Request", 0.74), ("Billing and Payments", 0.66), ("medium", 0.62), 0.12, []
    elif product:
        t, q, p, crit, hits = ("Request", 0.81), ("Product Support", 0.55), ("low", 0.58), 0.05, []
    else:
        t, q, p, crit, hits = ("Problem", 0.48), ("Technical Support", 0.31), ("medium", 0.49), 0.18, []
    return {
        "mode": "mock",
        "type": _task(TYPES, *t),
        "queue": _task(QUEUES, *q),
        "priority": _task(PRIORITIES, *p),
        "critical": {"score": crit, "threshold": 0.40, "fires": crit >= 0.40, "keyword_hits": hits},
        "neighbours": {"top": [], "explanation": {"type": "mock: no neighbour votes"}},
        "suggested_conf_thresholds": {"type": 0.55, "queue": 0.35, "priority": 0.45},
        "latency_ms": {"total": 0.1},
    }


def mock_search(text, k=5, preds=None, lang=None, exclude_ids=None, pool_ids=None) -> list[dict]:
    words = set(re.findall(r"\w{4,}", text.lower()))
    rows = []
    for tid, lg, ty, qu, pr, subj, ans in _PAST:
        overlap = len(words & set(re.findall(r"\w{4,}", (subj + " " + ans).lower())))
        rows.append({"ticket_id": tid, "similarity": round(min(0.96, 0.79 + 0.025 * overlap), 4), "language": lg,
                     "type": ty, "queue": qu, "priority": pr, "subject": subj, "answer": ans})
    rows.sort(key=lambda r: -r["similarity"])
    return rows[:k]
