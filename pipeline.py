"""SentinelDesk end-to-end: one call from raw ticket to everything the console shows.

    from pipeline import Pipeline
    p = Pipeline()                      # loads triage + retrieval once (wrap in st.cache_resource)
    out = p.handle(subject, body)

Order (README "Pipeline order"): mask PII -> detect language -> triage -> search -> gate ->
draft (LLM) -> verify -> final gate. Pre-draft escalations skip the LLM call.

The drafter is pluggable until the LLM is chosen:
    drafter(ticket_text, hits, lang) -> {"reply": str, "cited_ids": [str], "unsupported_asks": [str]}
Without one, safe tickets come back as DRAFT_PENDING.

  python pipeline.py                  # self-check on demo tickets
  python pipeline.py "subject" "body" # one ticket, printed as JSON
"""
from __future__ import annotations

import json
import sys
import time

from pii import mask
from retrieve import SIGMA, Retriever, top_similarity
from sentineldesk.language import detect_language
from sentineldesk.triage.predict import Triage
from sentineldesk.triage.text import build_text

try:  # PR #5 (verify step); the pipeline runs without it until it merges
    from sentineldesk.verify import verify
except ImportError:
    verify = None

ESCALATE, READY, PENDING = "ESCALATE", "AUTO_DRAFT_READY", "DRAFT_PENDING"
# ponytail: low queue confidence alone doesn't escalate (queue F1 ~0.5 would send most tickets to a
# human); it is shown as "routing uncertain" with the top-2 queues. Gate owner may tighten this.
ESCALATE_ON_LOW_CONF = ("type", "priority")


def gate(tri: dict, hits: list, draft: dict | None = None, checked: dict | None = None) -> list[str]:
    """Reasons to escalate (empty list = safe). Every rule has its own reason string for the UI."""
    reasons = []
    if tri["critical"]["fires"]:
        kw = ", ".join(tri["critical"]["keyword_hits"][:3])
        reasons.append(f"Critical incident: score {tri['critical']['score']:.2f} ≥ {tri['critical']['threshold']:.2f}"
                       + (f" (keywords: {kw})" if kw else ""))
    if tri["type"]["label"] == "Incident" and tri["priority"]["label"] == "high":
        reasons.append("High-priority Incident")
    for t in ESCALATE_ON_LOW_CONF:
        thr = tri["suggested_conf_thresholds"][t]
        if tri[t]["confidence"] < thr:
            reasons.append(f"Low {t} confidence: {tri[t]['confidence']:.2f} < {thr:.2f}")
    sim = top_similarity(hits)
    if sim < SIGMA:
        reasons.append(f"No historical precedent: best similarity {sim:.3f} < {SIGMA}")
    if draft and draft.get("unsupported_asks"):
        reasons.append("Unsupported request: " + "; ".join(draft["unsupported_asks"][:3]))
    if checked and not checked["passed"]:
        reasons.append("Draft failed verification: " + "; ".join(map(str, checked["failures"][:3])))
    return reasons


class Pipeline:
    def __init__(self, artifacts="artifacts", drafter=None):
        self.triage = Triage.load(artifacts)
        self.retriever = Retriever(embedder=self.triage.embedder)  # one e5-base model in memory
        self.drafter = drafter
        self.handle("warm-up", "VPN not working")  # first embedding call is ~1 s; pay it at startup

    def handle(self, subject, body) -> dict:
        lat, t0 = {}, time.perf_counter()

        def lap(name, t):
            lat[name] = round((time.perf_counter() - t) * 1000, 1)
            return time.perf_counter()

        t = time.perf_counter()
        subject, body = mask(subject or ""), mask(body or "")
        text = build_text(subject, body)  # same text builder as the index (\n, <br>)
        lang = detect_language(subject, body)
        t = lap("mask_lang", t)
        tri = self.triage.predict(subject, body)
        t = lap("triage", t)
        preds = {k: tri[k]["probs"] for k in ("type", "queue")}
        hits = self.retriever.search(text, k=5, preds=preds, lang=lang)
        t = lap("search", t)

        reasons, draft, checked = gate(tri, hits), None, None
        if not reasons and self.drafter is not None:
            draft = self.drafter(text, hits, lang)
            t = lap("draft", t)
            checked = verify(draft, hits, text, lang) if verify else None
            t = lap("verify", t)
            reasons = gate(tri, hits, draft, checked)
        status = ESCALATE if reasons else (READY if draft else PENDING)
        lat["total"] = round((time.perf_counter() - t0) * 1000, 1)

        q = sorted(tri["queue"]["probs"].items(), key=lambda x: -x[1])[:2]
        return {
            "status": status,  # ESCALATE -> "Escalate to On-Call Engineer"; AUTO_DRAFT_READY; DRAFT_PENDING
            "reasons": reasons,
            "language": lang,
            "masked_text": text,
            "triage": {k: {"label": tri[k]["label"], "confidence": tri[k]["confidence"]}
                       for k in ("type", "queue", "priority")},
            "routing_uncertain": tri["queue"]["confidence"] < tri["suggested_conf_thresholds"]["queue"],
            "queue_top2": [{"queue": n, "confidence": round(p, 3)} for n, p in q],
            "critical": tri["critical"],
            "neighbour_votes": tri["neighbours"].get("explanation", {}),
            "precedents": [{**h, "snippet": h["answer"][:240]} for h in hits],
            "best_similarity": top_similarity(hits),
            "draft": draft,
            "verify": checked,
            "verify_available": verify is not None,
            "triage_mode": tri["mode"],
            "latency_ms": lat,
        }


DEMO = [  # (subject, body, must escalate?)
    ("URGENT – VPN down", "Our VPN gateway has been down since 08:00. All remote clinicians are cut off from the "
     "EMR and telemedicine platform. Patients are waiting. Contact me at dr.lee@clinic.org or +1 617 555 0199.", True),
    ("", "What is a good recipe for lasagna?", True),
    ("Passwort zurücksetzen", "Hallo, ich habe mein Passwort vergessen und die Reset-Mail kommt nicht an. "
     "Können Sie mir helfen?", None),
    ("Invoice charged twice", "We were billed twice for the November subscription. Please refund one charge.", None),
]


def check():
    p = Pipeline()
    for subject, body, must in DEMO:
        out = p.handle(subject, body)
        tri = out["triage"]
        print(f"\n[{out['status']}] {subject or body[:40]!r}  lang={out['language']}  "
              f"{tri['type']['label']}/{tri['queue']['label']}/{tri['priority']['label']}  "
              f"best_sim={out['best_similarity']:.3f}  {out['latency_ms']['total']}ms")
        for r in out["reasons"]:
            print("   -", r)
        print("   top precedent:", out["precedents"][0]["ticket_id"], out["precedents"][0]["subject"][:60])
        assert "@" not in out["masked_text"] and "555" not in out["masked_text"], "PII leaked past mask()"
        if must is True:
            assert out["status"] == ESCALATE, f"should escalate: {subject or body}"
    off = p.handle("", "What is a good recipe for lasagna?")
    assert any(r.startswith("No historical precedent") for r in off["reasons"]), off["reasons"]
    print(f"\npipeline self-check OK (verify step {'on' if verify else 'not merged yet'}, drafter "
          f"{'on' if p.drafter else 'not configured'})")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(Pipeline().handle(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2, default=str))
    else:
        check()
