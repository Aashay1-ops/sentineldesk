"""Text-level guardrails for the console: PII counts, prompt-injection and sensitive-request lexicons.

Masking itself is the team's pii.mask(); this module only counts what it replaced. Every flag carries the
matched words, so the console can show *why* a ticket was routed to a person.
"""
from __future__ import annotations

import re

from pii import mask

PII_TAGS = {"<email>": "EMAIL", "<tel_num>": "PHONE", "<ip>": "IP", "<acc_num>": "ACCOUNT/CARD"}


def mask_with_counts(text: str) -> tuple[str, dict[str, int]]:
    """pii.mask(text) plus how many of each kind it replaced (dataset placeholders already in the text don't count)."""
    masked = mask(text or "")
    counts = {name: masked.count(tag) - (text or "").count(tag) for tag, name in PII_TAGS.items()}
    return masked, {k: v for k, v in counts.items() if v > 0}


# ---------------------------------------------------------------- prompt injection
_INJECTION = re.compile(
    r"(?i)("
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above|system)?\s*(?:instructions|rules|guidelines)"
    r"|system\s+prompt|developer\s+mode|jailbreak|you\s+are\s+now\s+(?:a|an|in)\b"
    r"|reveal\s+(?:your|the)\s+(?:instructions|prompt|rules)"
    r"|pretend\s+(?:to\s+be|you\s+are)|act\s+as\s+(?:a|an)\s+\w+\s+(?:without|with\s+no)"
    r"|promise\s+(?:me\s+)?(?:a\s+)?(?:full\s+)?refund\s+in\s+writing"
    r"|ignorier(?:e|en)?\s+(?:alle\s+)?(?:vorherigen|bisherigen|obigen)\s+(?:anweisungen|regeln)"
    r"|vergiss\s+(?:alle\s+)?(?:vorherigen\s+)?anweisungen|systemprompt"
    r"|</?\s*(?:system|assistant|instructions?)\s*>"
    r")"
)


def detect_injection(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in _INJECTION.finditer(text or "")})


# ---------------------------------------------------------------- sensitive requests
def _rx(words: list[str]) -> re.Pattern:
    return re.compile(r"(?i)(?<![\wäöüß])(" + "|".join(words) + r")(?![\wäöüß])")


# Money, contract and data-deletion asks: only a person may grant these, whatever a past answer said.
SENSITIVE_REQUESTS: dict[str, re.Pattern] = {
    "refund_compensation": _rx([
        r"refunds?", r"refunded", r"refunding", r"reimburse\w*", r"money\s+back", r"compensat\w+", r"credit\s+note",
        r"chargeback", r"service\s+credits?", r"(?:rück)?erstatt\w*", r"geld\s+zurück", r"entschädigung\w*", r"gutschrift\w*",
    ]),
    "pricing_discount": _rx([
        r"discounts?", r"price\s+quote", r"quotation", r"how\s+much\s+(?:does|is|would)", r"pricing", r"price\s+list",
        r"rabatt\w*", r"preis\w*", r"kostenvoranschlag", r"was\s+kostet",
    ]),
    "legal_sla_guarantee": _rx([
        r"sla", r"guarantee\w*", r"legal(?:ly)?", r"lawsuit", r"lawyer", r"attorney", r"liab\w+", r"penalt\w+", r"sue",
        r"garantie\w*", r"rechtlich\w*", r"anwalt", r"haftung", r"vertragsstrafe\w*", r"klage\w*",
    ]),
    "data_deletion": _rx([
        r"delete\s+(?:all\s+)?(?:my|our)\s+(?:data|account)", r"gdpr", r"right\s+to\s+(?:be\s+forgotten|erasure)",
        r"dsgvo", r"löschung\w*", r"daten\s+löschen",
    ]),
}


def match_lexicon(text: str, lexicon: dict[str, re.Pattern]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for cat, pat in lexicon.items():
        found = sorted({m.group(0).lower() for m in pat.finditer(text or "")})
        if found:
            hits[cat] = found
    return hits


# Figures a reply must never introduce unless the ticket or a precedent contains them.
FIGURE = re.compile(
    r"(?i)(?:[$€£]\s?\d[\d.,]*|\d[\d.,]*\s?(?:%|€|eur|usd|dollars?|euros?|prozent)|"
    r"\d+\s?(?:business\s+|working\s+)?(?:hours?|days?|weeks?|months?|stunden|tagen?|wochen|monaten?|werktagen?))"
)
