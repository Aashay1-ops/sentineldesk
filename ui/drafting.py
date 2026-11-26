"""Offline reply drafting for the console (stand-in until the team's draft + verify are wired in).

The team's LLM draft is sentineldesk/draft.py (Gemini) and verify is PR #5; until both are wired into the
console, this module drafts offline: a deterministic extractive draft from the closest same-language
precedent, or a fixed holding reply when there is nothing to ground on (or the ticket is adversarial).

Every draft goes through verify_draft (README: "the verify step must pass, or the ticket escalates"):
citations must be retrieved tickets, no figures or commitments absent from the evidence, no personal data,
no refund talk unless it is deferred to a person, and the reply must be in the requester's language.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from pii import mask
from sentineldesk.language import detect_language
from ui.guardrails import FIGURE, SENSITIVE_REQUESTS

CITATION = re.compile(r"TKT-\d{5}")
COMMITMENT = re.compile(
    r"(?i)(guarantee\w*|we\s+will\s+(?:refund|reimburse|compensate|credit)|full\s+refund|compensation\s+of|"
    r"free\s+of\s+charge|garantier\w*|wir\s+(?:werden\s+)?(?:erstatten|entschädigen)|kostenlos|kostenfrei)"
)
DEFERRED = re.compile(r"(?i)review|prüf|weitergeleitet|forwarded|passed")

HOLDING = {
    "en": {
        "review": "Dear customer,\n\nthank you for your message. We have received your request and passed it to the responsible team, "
                  "who will review it and get back to you. To help us move quickly, please reply with any relevant details, such as "
                  "the affected systems, exact error messages and when the issue started.\n\nBest regards,\nSentinelDesk Support",
        "oncall": "Dear customer,\n\nthank you for reporting this. We have classified it as a high-priority incident and our on-call "
                  "engineering team has been alerted. To speed up the investigation, please send us the affected systems, exact error "
                  "messages, relevant logs and the time the problem started.\n\nBest regards,\nSentinelDesk Support",
    },
    "de": {
        "review": "Sehr geehrte Damen und Herren,\n\nvielen Dank für Ihre Nachricht. Wir haben Ihre Anfrage erhalten und an das "
                  "zuständige Team weitergeleitet, das sie prüfen und sich bei Ihnen melden wird. Damit wir schnell helfen können, "
                  "senden Sie uns bitte relevante Details wie die betroffenen Systeme, genaue Fehlermeldungen und den Zeitpunkt, "
                  "seit dem das Problem besteht.\n\nMit freundlichen Grüßen\nSentinelDesk Support",
        "oncall": "Sehr geehrte Damen und Herren,\n\nvielen Dank für Ihre Meldung. Wir haben den Vorfall als dringend eingestuft und "
                  "unser Bereitschaftsteam informiert. Um die Analyse zu beschleunigen, senden Sie uns bitte die betroffenen Systeme, "
                  "genaue Fehlermeldungen, relevante Protokolle und den Zeitpunkt, seit dem das Problem besteht.\n\n"
                  "Mit freundlichen Grüßen\nSentinelDesk Support",
    },
}
GREETING = {"en": "Dear customer,", "de": "Sehr geehrte Damen und Herren,"}
THANKS = {"en": "thank you for contacting us.", "de": "vielen Dank für Ihre Nachricht."}
CLOSING = {"en": "Best regards,\nSentinelDesk Support", "de": "Mit freundlichen Grüßen\nSentinelDesk Support"}


@dataclass
class DraftResult:
    reply: str
    mode: str                       # "extractive" | "holding"
    cited_ticket_ids: list[str]
    unsupported_requests: list[str] = field(default_factory=list)
    grounding: str = "partial"
    internal_note: str = ""
    issues: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)  # draft / verify ms and LLM model name (pipeline strip)
    trace: dict = field(default_factory=dict)    # LLM path: prompt (system, user), raw JSON, verify checks


# ---------------------------------------------------------------- resolution cleanup
_BRACKET_PH = re.compile(r"\[([A-Za-zÄÖÜäöü_ /]{2,30})\]|\{([a-z_ ]{2,20})\}")
_SALUTATION = re.compile(
    r"^\s*(?:(?:dear|hello|hi|sehr\s+geehrte\S*|liebe\S*|hallo|guten\s+tag)\b[^,\n]{0,40}|<[a-z_]+>)\s*[,!]\s*", re.I)
_SIGNOFF = re.compile(
    r"\n?\s*(?:best|kind|warm)\s+regards\b.*|\n?\s*sincerely\b.*|\n?\s*(?:mit\s+)?(?:freundlichen|beste[n]?|viele[n]?|herzliche[n]?)\s+grüße?n?\b.*",
    re.I | re.S)
_THANKS = re.compile(r"^(?:thank\s+you|thanks|we\s+appreciate|vielen\s+dank|danke|wir\s+danken|wir\s+bedanken)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")
_ABBREV = re.compile(r"(?i)\b(?:nr|no|z\.\s?b|bzw|ca|dr|e\.g|i\.e|etc|vs|inkl|ggf|usw|tel)\.$")
_PLACEHOLDER = re.compile(r"<[a-z_]+>")


def _sentences(text: str) -> list[str]:
    """Split on . ! ? but not after abbreviations ("Konto-Nr. <acc_num> ..." stays one sentence)."""
    out: list[str] = []
    for part in (s.strip() for s in _SENT.split(text) if s.strip()):
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def clean_resolution(answer: str) -> str:
    """Literal '\\n' -> line break (4.3% of KB rows), [Name]/{tel_num} -> <name>/<tel_num>, strip salutation/sign-off."""
    s = (answer or "").replace("\\n", "\n")
    s = _BRACKET_PH.sub(lambda m: "<" + re.sub(r"[\s/]+", "_", (m.group(1) or m.group(2)).strip().lower()) + ">", s)
    s = _SIGNOFF.sub("", _SALUTATION.sub("", s.strip())).strip()
    return s[:1].upper() + s[1:] if s else s


def _answer_lang(p: dict) -> str:
    return p.get("answer_language") or detect_language("", p.get("answer") or "")


# ---------------------------------------------------------------- drafts
def holding_reply(language: str, status: str) -> DraftResult:
    kind = "oncall" if status == "ESCALATE_ON_CALL" else "review"
    return DraftResult(reply=HOLDING.get(language, HOLDING["en"])[kind], mode="holding", cited_ticket_ids=[], grounding="none")


def extractive_draft(language: str, precedents: list[dict], status: str) -> DraftResult:
    same_lang = [p for p in precedents if p.get("answer") and _answer_lang(p) == language]
    if not same_lang:
        d = holding_reply(language, status)
        d.internal_note = ("No same-language precedent for an offline extractive draft; cross-lingual precedents: "
                           + ", ".join(p["ticket_id"] for p in precedents))
        return d
    best = same_lang[0]
    sentences = _sentences(clean_resolution(best["answer"]).replace("\n", " "))
    keep = [s for s in sentences if not _PLACEHOLDER.search(s) and not _THANKS.match(s)][:4]
    keep = keep or [s for s in sentences if not _PLACEHOLDER.search(s)][:3] or sentences[:3]
    body = " ".join(keep)
    lang = language if language in GREETING else "en"
    reply = f"{GREETING[lang]}\n\n{THANKS[lang]} {body} [{best['ticket_id']}]\n\n{CLOSING[lang]}"
    if status == "ESCALATE_ON_CALL":
        reply = HOLDING[lang]["oncall"].rsplit("\n\n", 1)[0] + f"\n\n{body} [{best['ticket_id']}]\n\n{CLOSING[lang]}"
    return DraftResult(reply=reply, mode="extractive", cited_ticket_ids=[best["ticket_id"]], grounding="partial",
                       internal_note="Offline extractive draft from the closest same-language resolution.")


# ---------------------------------------------------------------- verification
def verify_draft(draft: DraftResult, precedents: list[dict], ticket_text: str, language: str) -> DraftResult:
    retrieved = {p["ticket_id"] for p in precedents}
    in_text = set(CITATION.findall(draft.reply)) | {t for t in retrieved if t in draft.reply}
    issues = []

    bogus = sorted(in_text - retrieved)
    if bogus:
        issues.append(f"cited non-retrieved ticket(s) {', '.join(bogus)} (removed)")
        for b in bogus:
            draft.reply = re.sub(rf"\s?\[?{b}\]?", "", draft.reply)
    draft.cited_ticket_ids = sorted(in_text & retrieved)
    if draft.mode != "holding" and not draft.cited_ticket_ids:
        issues.append("no precedent citations in a grounded draft")

    evidence = (ticket_text + "\n" + "\n".join(p.get("answer") or "" for p in precedents)).lower()
    ungrounded = sorted({f for f in FIGURE.findall(draft.reply) if f.lower() not in evidence})
    if ungrounded and draft.mode != "holding":
        issues.append("figures not found in evidence: " + ", ".join(ungrounded))

    commitments = sorted({m.group(0).lower() for m in COMMITMENT.finditer(draft.reply)})
    unbacked = [c for c in commitments if c not in evidence]
    if unbacked:
        issues.append("commitment language not backed by precedent: " + ", ".join(unbacked))

    # A past refund was granted for a different case, so refund talk is fine only when deferred to a person.
    if SENSITIVE_REQUESTS["refund_compensation"].search(draft.reply) and not DEFERRED.search(draft.reply):
        issues.append("discusses refunds/compensation without deferring to a person")

    if mask(draft.reply) != draft.reply:
        issues.append("reply contains personal data (email, phone, IP or account number)")

    reply_lang = detect_language("", draft.reply)
    if reply_lang != language:
        issues.append(f"reply language {reply_lang} != requester language {language}")

    draft.issues = issues
    return draft


def llm_draft(ticket_text, language, precedents, status, llm, check) -> DraftResult:
    """Team draft (sentineldesk/draft.py) + team verify (sentineldesk/verify.py). A draft that fails verify is
    never shown (README: verify must pass, or the ticket escalates): the holding reply replaces it and the
    failures become issues, which the gate turns into DRAFT_GUARD. LLM unavailable -> offline extractive draft."""
    t0 = time.perf_counter()
    trace: dict = {}
    try:
        from sentineldesk.draft import build_prompt  # same pure function draft() uses: the exact prompt sent

        trace["prompt"] = build_prompt(ticket_text, precedents, language)
        out = llm(ticket_text, precedents, language)
    except Exception as e:  # DraftError (no key, quota, timeout) or anything else: never crash the console
        d = verify_draft(extractive_draft(language, precedents, status), precedents, ticket_text, language)
        d.internal_note = f"LLM unavailable ({type(e).__name__}: {str(e)[:160]}) - offline extractive draft instead."
        d.timings = {"draft": (time.perf_counter() - t0) * 1000, "model": "LLM unavailable"}
        d.trace = {**trace, "error": f"{type(e).__name__}: {e}"}
        return d
    t1 = time.perf_counter()
    res = check(out, precedents, ticket_text, language)
    trace.update(raw=out, checks=res.get("checks", []), failures=res.get("failures", []))
    timings = {"draft": (t1 - t0) * 1000, "verify": (time.perf_counter() - t1) * 1000, "model": out.get("model")}
    model = f"{out.get('model', 'LLM')} in {out.get('latency_ms', '?')} ms"
    asks = [str(a) for a in out.get("unsupported_asks") or []]
    if res["passed"]:
        return DraftResult(reply=out["reply"], mode="llm", cited_ticket_ids=list(out["cited_ids"]), grounding="full",
                           internal_note=f"Drafted by {model}; passed all verify checks.", timings=timings,
                           trace=trace)
    d = holding_reply(language, status)
    d.trace = trace
    d.issues = [f"{f['check']}: {f['detail']}" for f in res["failures"] if f["code"] != "UNSUPPORTED_ASK"]
    d.unsupported_requests = asks
    d.internal_note = f"LLM draft ({model}) failed verify, so it is hidden; holding reply shown instead."
    d.timings = timings
    return d


def draft_reply(ticket_text, language, precedents, status, has_precedent, injection=False,
                llm=None, check=None) -> DraftResult:
    if llm and check and status == "AUTO_DRAFT_READY" and has_precedent and not injection:
        # ponytail: the LLM only drafts tickets the gate already cleared (saves quota; reviewed/escalated
        # tickets keep the offline draft). Upgrade path: also draft HUMAN_REVIEW tickets for the agent.
        return llm_draft(ticket_text, language, precedents, status, llm, check)
    if injection:
        # Adversarial input: nothing is generated from this text - fixed holding reply for the human reviewer.
        d = holding_reply(language, status)
        d.internal_note = "Prompt-injection attempt detected - holding reply only, nothing generated from this text."
        return verify_draft(d, precedents, ticket_text, language)
    if not has_precedent:
        d = holding_reply(language, status)
        d.internal_note = "No historical precedent - holding reply only (nothing to ground on)."
        return verify_draft(d, precedents, ticket_text, language)
    return verify_draft(extractive_draft(language, precedents, status), precedents, ticket_text, language)
