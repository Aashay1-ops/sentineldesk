"""PII guardrail: mask personal data before any text leaves the machine (LLM API) or reaches the index.

Placeholders match the dataset's own anonymisation (<email>, <tel_num>, <acc_num>, <ip>), so masked live
tickets look like the knowledge base and the verify step can check them.
"""
import re

_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
_PATTERNS = [  # most specific first, so the phone pattern can't eat their digits
    ("<email>", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<acc_num>", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")),  # IBAN
    ("<acc_num>", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),                                        # card
    ("<ip>", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    (r"\1<acc_num>", re.compile(r"(?i)(\b(?:acct|account|konto|kontonummer|kundennummer|customer\s*(?:id|no))"
                                r"[\s#:.-]*)(?=[A-Z-]*\d)[A-Z0-9-]{4,}\b")),
]
_PHONE = re.compile(r"(?<![\w.])(?:\+|00)?\(?\d[\d\s()/.-]{5,}\d(?!\w)")

_CANON = {
    "<name>": "name yourname ihrname customername customer kundenname",
    "<tel_num>": "telnum telnumber telnummer telefonnummer phonenumber phone",
    "<acc_num>": "accnum accountnumber accountnummer kontonummer kontonr kundennummer accountno",
    "<email>": "email emailaddress emailadresse",
}
_LOOKUP = {k: tag for tag, keys in _CANON.items() for k in keys.split()}
_BRACKET = re.compile(r"[\[{]\s*([A-Za-zÄÖÜäöüß _-]{2,30}?)\s*[\]}]")


def _phone(m):
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    return "<tel_num>" if 7 <= digits <= 15 and not _DATE.match(s.strip()) else s


def mask(text: str) -> str:
    """Text with emails, account/card/IBAN numbers, IPs and phone numbers replaced by placeholders."""
    if not text:
        return text or ""
    for tag, rx in _PATTERNS:
        text = rx.sub(tag, text)
    return _PHONE.sub(_phone, text)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """(masked text, count of each placeholder added)."""
    masked = mask(text)
    counts: dict[str, int] = {}
    for tag in ("<email>", "<acc_num>", "<ip>", "<tel_num>"):
        n = masked.count(tag) - (text or "").count(tag)
        if n > 0:
            counts[tag] = n
    return masked, counts


def normalize_placeholders(text: str) -> str:
    """[Name], {tel_num}, [Telefonnummer] ... -> <name>, <tel_num>, ... (unknown brackets are kept)."""
    def sub(m):
        return _LOOKUP.get(re.sub(r"[\s_-]", "", m.group(1)).lower(), m.group(0))
    return _BRACKET.sub(sub, text or "")
