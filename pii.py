"""PII guardrail: mask emails, phones, IPs and account/card numbers before text reaches
triage, retrieval or any external LLM. Placeholders match the dataset's own anonymization
(<email>, <tel_num>, <acc_num>), so masked live tickets look like the training data.

ponytail: regex only. Personal names are NOT masked (regex can't find them reliably);
upgrade path is a spaCy/NER pass that maps PERSON -> <name>.

  python pii.py      # self-check
"""
import re

_DIGITS = re.compile(r"\d")
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[./]\d{1,2}[./]\d{2,4}$")

# Order matters: the most specific patterns run first, so the phone regex can't eat their digits.
_RULES = [
    ("<email>", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<ip>", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("<ip>", re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){4,7}[0-9a-f]{1,4}(?![\w:])|\b[0-9a-f]{1,4}::[0-9a-f:]*")),
    # IBAN: 2 letters, 2 check digits, then 11-30 alphanumerics, optionally grouped by 4.
    ("<acc_num>", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")),
    # Card numbers: 13-19 digits, optionally in groups split by spaces or dashes.
    ("<acc_num>", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    # Account numbers introduced by a keyword (en/de): keep the keyword, mask the number.
    (r"\1<acc_num>", re.compile(r"(?i)(\b(?:account|acct|konto|kontonummer|kundennummer|customer)"
                                r"(?: (?:no|nr|number|nummer|id))?[.:#]? ?)(?=[A-Z-]*\d)[A-Z0-9-]{5,}\b")),
]
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s()/.-]{5,}\d(?!\w)")


def _phone(m):
    s = m.group()
    n = len(_DIGITS.findall(s))
    # 7-15 digits (E.164 max is 15) and not a date like 2026-11-21 or 21.11.2026.
    return "<tel_num>" if 7 <= n <= 15 and not _DATE.match(s.strip()) else s


def mask(text):
    if not text:
        return text or ""
    for tag, rx in _RULES:
        text = rx.sub(tag, text)
    return _PHONE.sub(_phone, text)


# Data audit #8: 12.4% of answers use [Name] / {tel_num} / [Telefonnummer] instead of <name> / <tel_num>.
# Run on retrieved answers before the drafting LLM, so the verify step only has to allow <...> placeholders.
_BRACKET = re.compile(r"[\[{]\s*([A-Za-zÄÖÜäöüß _.'’-]{2,30}?)\s*[\]}]")
# [Your Name] / [Ihr Name] / [Dein Name] are left alone on purpose: in answers they are usually the
# AGENT's sign-off slot, and mapping them to <name> would make the LLM address the customer by it.
_CANON = {
    "<name>": {"name", "customername", "customersname", "customer", "kundenname"},
    "<tel_num>": {"telnum", "telnumber", "telnummer", "telefonnummer", "phonenumber", "phone"},
    "<acc_num>": {"accnum", "accountnumber", "accountnummer", "kontonummer", "kontonr", "kundennummer", "accountno"},
    "<email>": {"email", "emailaddress", "emailadresse"},
}
_LOOKUP = {k: tag for tag, keys in _CANON.items() for k in keys}


def normalize_placeholders(text):
    """[Name], {tel_num}, [Telefonnummer], ... -> <name>, <tel_num>, <acc_num>. Unknown brackets are kept.
    Non-strings (None, NaN from pandas) return ""."""
    if not isinstance(text, str):
        return ""

    def sub(m):
        return _LOOKUP.get(re.sub(r"[\s_.'’-]", "", m.group(1)).lower(), m.group(0))
    return _BRACKET.sub(sub, text)


if __name__ == "__main__":
    for src, want in {
        "Dear [Name], call [Tel-Nummer] or {tel_num}.": "Dear <name>, call <tel_num> or <tel_num>.",
        "Hallo [Name], Konto [Kontonummer], Grüße [Ihr Name]": "Hallo <name>, Konto <acc_num>, Grüße [Ihr Name]",
        "See [Phone Number] and [Acc_Num], [Konto-Nr.] [Customer's Name] [Tel.Nummer]":
            "See <tel_num> and <acc_num>, <acc_num> <name> <tel_num>",
        "Best regards, [Your Name]": "Best regards, [Your Name]",
        "Step [1] of the guide [Settings > Users]": "Step [1] of the guide [Settings > Users]",
        "Already <name> and <tel_num>": "Already <name> and <tel_num>",
    }.items():
        assert normalize_placeholders(src) == want, f"{src!r} -> {normalize_placeholders(src)!r}"

    must_mask = {
        "mail anna.schmidt@klinik-berlin.de please": "mail <email> please",
        "call +49 30 1234567 now": "call <tel_num> now",
        "call (617) 555-0199.": "call <tel_num>.",
        "Tel: 030/12345678": "Tel: <tel_num>",
        "gateway 10.0.4.21 down": "gateway <ip> down",
        "host 2001:db8:85a3:0:0:8a2e:370:7334 down": "host <ip> down",
        "IBAN DE89 3704 0044 0532 0130 00 bitte": "IBAN <acc_num> bitte",
        "card 4111 1111 1111 1111 charged": "card <acc_num> charged",
        "card 4111-1111-1111-1111": "card <acc_num>",
        "Kundennummer: 88213345 gesperrt": "Kundennummer: <acc_num> gesperrt",
        "account no. AC-99812 locked": "account no. <acc_num> locked",
    }
    must_keep = [
        "Windows 10 error 404 after update 2.4.1",
        "since 2026-11-21 at 10:30:00",
        "seit dem 21.11.2026 um 08:00",
        "invoice of 49.99 EUR, 3 users, port 8080",
        "Dear <name>, call <tel_num> about <acc_num>",  # dataset placeholders stay intact
        "",
    ]
    for src, want in must_mask.items():
        assert mask(src) == want, f"{src!r} -> {mask(src)!r}, want {want!r}"
    for src in must_keep:
        assert mask(src) == src, f"false positive: {src!r} -> {mask(src)!r}"
    assert mask(None) == ""
    print(f"pii.py OK: {len(must_mask)} masked, {len(must_keep)} left alone")
