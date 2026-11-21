"""⑤ VERIFY: deterministic rules that check the LLM's draft before anyone sees it.

    from sentineldesk.verify import verify
    result = verify(draft, retrieved, ticket_text, lang)
    if not result["passed"]:
        ...  # escalate: never show the draft; result["failures"] lists every reason

Inputs
    draft        {"reply": str, "cited_ids": [str], "unsupported_asks": [str]}  (the LLM's JSON)
    retrieved    search() output: [{"ticket_id", "answer", "language", ...}, ...]
    ticket_text  the customer's ticket, already PII-masked (subject + "\\n" + body)
    lang         "en" or "de", from sentineldesk.language

Checks (all run every time, so the agent sees every failure)
    format            reply non-empty, <= MAX_WORDS words and MAX_CHARS characters; list fields   VERIFY_FAILED
    citations         cited ids (field + inline TKT-xxxxx) all retrieved, at least one            VERIFY_FAILED
    numbers           amounts / % / quantities with a unit must appear, same unit, in a retrieved  VERIFY_FAILED
                      answer; plain numbers, dates and times in a retrieved answer or the ticket
    placeholders      <placeholders> come from the sources; no invented email / phone / account /  VERIFY_FAILED
                      link (text quoted verbatim from a source is not counted as invented)
    language          the whole reply and every full sentence are in the ticket's language        VERIFY_FAILED
    unsupported_asks  the LLM listed nothing it could not ground                                  UNSUPPORTED_ASK

Pure Python: re, difflib, decimal, the repo's pii module and sentineldesk.language. Typically a few ms per draft.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from pii import mask, normalize_placeholders
from sentineldesk.language import detect_language_with_confidence

MAX_WORDS = 250
MAX_CHARS = 3000          # one 40k-character "word" must not slip past MAX_WORDS or slow the checks down
MIN_WORDS_FOR_LANG = 8    # alphabetic words; the detector is unreliable below this
MIN_WORDS_PER_SENTENCE = 6
SENTENCE_LANG_CONF = 0.7  # a full sentence in the other language at this confidence fails the reply
MAX_SOURCE_CHARS = 20000   # ticket / each answer are cut here: pasted CSV rows must not stall the gate
MAX_DIGITS = 30           # longer numbers are not read (and so never match): no overflow, no int-size limits

VERIFY_FAILED = "VERIFY_FAILED"
UNSUPPORTED_ASK = "UNSUPPORTED_ASK"

_TKT = re.compile(r"\bTKT-\d{5}\b")
_PH_ANY = re.compile(r"<[A-Za-z][A-Za-z_ -]{0,30}>")   # any <placeholder>, any case
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\]\)]+", re.I)
# any host.tld, Unicode (IDN, lookalike letters) included; file names and "update.Then" are not links
_DOMAIN = re.compile(r"(?<![\w@.])(?:[^\W_](?:[\w-]{0,61}[^\W_])?\.){1,4}[^\W\d_]{2,24}\b(?:/[^\s<>()\]]*)?")
_FILE_EXT = set("exe pdf zip txt doc docx xls xlsx csv log json xml png jpg jpeg gif svg js ts py ini cfg conf yaml "
                "yml dll msi dmg apk iso bat sh ps1 md html htm tmp bak sql db pst ost eml msg".split())
_AT_B, _DOT_B = r"\s*[(\[{]\s*(?:at|ät)\s*[)\]}]\s*", r"\s*[(\[{]\s*(?:dot|punkt)\s*[)\]}]\s*"
_OBF_EMAIL = re.compile(   # name(at)firma.de, name [at] firma [dot] de, name at firma dot com
    rf"\b[\w.+-]{{1,64}}{_AT_B}[\w-]{{1,63}}(?:(?:{_DOT_B}|\.)[\w-]{{1,63}}){{1,4}}"
    rf"|\b[\w.+-]{{1,64}}\s+(?:at|ät)\s+[\w-]{{1,63}}(?:\s+(?:dot|punkt)\s+[\w-]{{1,63}}){{1,4}}\b", re.I)
_ALPHA_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")

# ---------------------------------------------------------------- units (en + de -> one canonical name)
_TIME_UNIT_WORDS = {
    "second": "second seconds sec secs sekunde sekunden",
    "minute": "minute minutes min mins minute minuten",
    "hour": "hour hours hr hrs h std stunde stunden",
    "day": "day days tag tage tagen tages werktag werktage werktagen werktages arbeitstag arbeitstage arbeitstagen "
           "arbeitstages geschäftstag geschäftstage geschäftstagen kalendertag kalendertage kalendertagen",
    "week": "week weeks woche wochen",
    "month": "month months monat monate monaten monats",
    "year": "year years yr yrs jahr jahre jahren jahres",
}
_UNIT_WORDS = {
    **_TIME_UNIT_WORDS,
    "times": "times mal fach fold",   # not "time": "one-time password" is not a quantity
    "data_kb": "kb", "data_mb": "mb", "data_gb": "gb", "data_tb": "tb",
    "user": "user users nutzer benutzer benutzerin",
    "seat": "seat seats platz plätze",
    "license": "license licenses licence licences lizenz lizenzen",
    "device": "device devices gerät geräte",
    "weekend": "weekend weekends wochenende wochenenden",
    "quarter": "quarters quartal quartale quartalen",
    "cycle": "cycle cycles zyklus zyklen abrechnungszeitraum abrechnungszeiträume",
    "period": "period periods zeitraum zeiträume",
    "payment": "payment payments installment installments instalment instalments rate raten zahlung zahlungen",
}
_UNITS = {w: canon for canon, words in _UNIT_WORDS.items() for w in words.split()}
_TIME_UNITS = {w for words in _TIME_UNIT_WORDS.values() for w in words.split()}
_FILLERS = ("more additional extra full further business working calendar weitere weiteren zusätzliche "
            "zusätzlichen volle vollen ganze ganzen").split()
# words that end the search for a unit: "Option 2 for business users" is not "2 users"
_STOP = ("a an the of to for in on at by per with from and or than as is are was be this that these your our "
         "der die das den dem des ein eine einen einem einer eines für von zu zum zur in im am an auf mit pro je "
         "und oder als ist sind ihr ihre ihren unser unsere").split()
_CURRENCY = {"€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "$": "USD", "usd": "USD", "dollar": "USD",
             "dollars": "USD", "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "pfund": "GBP",
             "chf": "CHF", "franken": "CHF", "franc": "CHF", "francs": "CHF", "cent": "CENT", "cents": "CENT"}
_PCT_WORDS = ("%", "percent", "per cent", "prozent", "pct")
_MAGNITUDE = {"k": 10**3, "tsd": 10**3, "thousand": 10**3, "tausend": 10**3, "hundert": 100, "hundred": 100, "m": 10**6, "mio": 10**6,
              "million": 10**6, "millions": 10**6, "millionen": 10**6, "bn": 10**9, "mrd": 10**9,
              "billion": 10**9, "billions": 10**9, "milliarde": 10**9, "milliarden": 10**9}
# words after a number that scale it; single letters (k, m) only directly after a money amount
_MAG_WORD_RE = (r"\s?(?:mio|mrd|tsd)\.?|\s?bn\b"
                r"|\s(?:million|millions|millionen|milliarden?|billion|billions|thousand|tausend|hundert|hundred)\b")
_MAG_CUR_RE = rf"(?:{_MAG_WORD_RE}|[kKmM]\b)"
# billing period after an amount: "€20 per month", "20 € pro Jahr", "€20 monthly", "€20 a year" (-> "1 year")
_RATE_ADV = {"monthly": "month", "monatlich": "month", "annually": "year", "yearly": "year", "jährlich": "year",
             "weekly": "week", "wöchentlich": "week", "daily": "day", "täglich": "day", "hourly": "hour",
             "stündlich": "hour", "p.a.": "year", "p. a.": "year"}


def _alt(words) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_UNIT_RE, _FILL_RE, _STOP_RE = _alt(_UNITS), _alt(_FILLERS), _alt(_STOP)
_TIME_RE = _alt(_TIME_UNITS)
_CUR_SYM_RE, _CUR_WORD_RE = r"€|\$|£", _alt(k for k in _CURRENCY if k.isalpha())
_PCT_RE = _alt(_PCT_WORDS)
_WORD = r"[A-Za-zÄÖÜäöüß]+"
_GAP = rf"(?:[\s-]+(?:{_FILL_RE}))*[\s-]?"   # filler words only (between a number and %, currency)
# up to two words between a number and its unit, unless one of them is a stop word ("2 free months")
_GAP_ANY = rf"(?:(?:[\s-]+(?!(?:{_STOP_RE})\b){_WORD}\b){{1,3}}[\s-]+|[\s-]?)"
_NUM = r"(?:\d{1,3}(?:[ \u00a0\u202f'\u2019]\d{3}){1,6}(?:[.,]\d{1,12})?(?!\d)|\d{1,30}(?:[.,]\d{1,30}){0,6})(?:[.,]-)?"  # bounded: linear time on long "1,1,1," runs

# ---------------------------------------------------------------- number words (en + de)
_EN_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_EN_SCALE = {"hundred": 100, "thousand": 1000, "dozen": 12, "million": 10**6, "billion": 10**9}
_EN_WORDS = set(_EN_SMALL) | set(_EN_TENS) | set(_EN_SCALE)
_DE_ARTICLES = {"ein", "eine", "einen", "einem", "einer", "eines"}  # "one" only before a time unit
_DE_SMALL = {"null": 0, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
             "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
             "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_DE_TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
            "siebzig": 70, "achtzig": 80, "neunzig": 90}
_DE_ADJ_UNITS = {"tägig": "Tage", "wöchig": "Wochen", "monatig": "Monate", "jährig": "Jahre",
                 "stündig": "Stunden", "minütig": "Minuten"}


def _en_value(phrase: str) -> float | int | None:
    half = 0.0
    p = phrase.lower()
    if re.search(r"[\s-]+and[\s-]+a[\s-]+half$", p):  # "one and a half" = 1.5
        p, half = re.sub(r"[\s-]+and[\s-]+a[\s-]+half$", "", p), 0.5
    toks = [t for t in re.split(r"[\s-]+", p) if t and t != "and"]
    if toks and toks[0] in ("a", "an"):
        toks = toks[1:]
        if toks and toks[0] in _EN_SCALE:  # "a hundred" = 100; otherwise the article is just dropped
            toks = ["one"] + toks
    total = cur = 0
    for t in toks:
        if t in _EN_SMALL:
            cur += _EN_SMALL[t]
        elif t in _EN_TENS:
            cur += _EN_TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t in ("thousand", "million", "billion"):
            total, cur = total + (cur or 1) * _EN_SCALE[t], 0
        elif t == "dozen":
            cur = (cur or 1) * 12
        else:
            return None
    return (total + cur + half) if toks else None


def _de_value(word: str) -> float | int | None:
    """A complete German number word ("vierundzwanzig", "zweihundert"); anything else (einige, keine) is None."""
    w = word.lower()
    if w == "anderthalb":
        return 1.5
    if w.endswith("einhalb") and len(w) > 7:  # zweieinhalb = 2.5, eineinhalb = 1.5
        head = w[:-7]
        base = 1 if head == "ein" else _de_value(head)
        return None if base is None else base + 0.5
    if w in _DE_SMALL:
        return _DE_SMALL[w]
    if w in _DE_TENS:
        return _DE_TENS[w]
    for scale, mult in (("tausend", 1000), ("hundert", 100)):
        if scale in w:
            head, tail = w.split(scale, 1)
            h = 1 if head in ("", "ein") else _de_value(head)
            t = _de_value(tail.removeprefix("und")) if tail else 0
            return None if h is None or t is None else h * mult + t
    if "und" in w:
        ones, tens = w.split("und", 1)
        o = 1 if ones == "ein" else _DE_SMALL.get(ones)
        if o is not None and 1 <= o <= 9 and tens in _DE_TENS:
            return o + _DE_TENS[tens]
    return None


_EN_RUN = re.compile(rf"\b(?:(?:a|an)[\s-]+)?(?:{_alt(_EN_WORDS)})(?:[\s-]+(?:and[\s-]+)?(?:{_alt(_EN_WORDS)}))*"
                     rf"(?:[\s-]+and[\s-]+a[\s-]+half)?\b", re.I)
_DE_COMPOUND_UNIT = r"[a-zäöüß]{2,}?(?:tagen|tage|tages|stunden|wochen|monaten|monate|jahren|jahre)"
_FOLLOWED_BY_UNIT = re.compile(rf"{_GAP_ANY}(?:{_UNIT_RE}|{_CUR_WORD_RE}|{_PCT_RE}|{_DE_COMPOUND_UNIT})\b|\s?(?:{_CUR_SYM_RE}|%)"
                               rf"|(?:{_MAG_WORD_RE})", re.I)
_HALF = re.compile(rf"\b(?:half|halbe[nrs]?|halb)[\s-]+(?:(?:an?|ein(?:e[nmrs]?)?)\s+)?(?={_GAP}(?:{_TIME_RE})\b)", re.I)
_ARTICLE_TIME = re.compile(rf"\b(?:a|an|ein|eine|einen|einem|einer|eines)(?=\s+{_GAP}(?:{_TIME_RE})\b)", re.I)
_DE_ADJ = re.compile(rf"\b(\d+|{_WORD}?)-?({_alt(_DE_ADJ_UNITS)})(?:e[nmrs]?)?\b", re.I)
_FRACTION = {"half": "0.5", "hälfte": "0.5", "halbe": "0.5", "halben": "0.5", "halber": "0.5",
             "quarter": "0.25", "viertel": "0.25"}
# "double the refund" / "doppelter Betrag" promise a multiple; "double charge" (a billing error) does not
_MULTIPLE = re.compile(r"\b(?:double|twice|triple)\s+(?:the|your|our)\s+(?:amount|fee|price|refund|credit|sum|cost)s?\b"
                       r"|\b(?:doppelte[nrs]?|dreifache[nrs]?)\s+(?:betrag|summe|preis|gebühr|erstattung|gutschrift)\b",
                       re.I)
_FRACTION_RE = re.compile(rf"\b(?:{_alt(_FRACTION)})\b", re.I)


def _convert_number_words(text: str, unreadable: list[str]) -> str:
    """Rewrite quantities written in words as digits, so the patterns below can read them.
    "twenty-four hours" -> "24 hours", "an hour" -> "1 hour", "half an hour" -> "0.5 hour",
    "zweiwöchige" -> "2 Wochen", "vierundzwanzig Stunden" -> "24 Stunden". Number words that are not
    quantities ("one of our agents", "einige", "keine") are left alone. An English quantity in words that
    can't be read is recorded in `unreadable`, so the numbers check fails instead of skipping it."""
    def de_adj(m):
        prefix, value = m.group(1), None
        if prefix.isdigit():
            value = prefix
        elif prefix.lower() in ("halb", "halbe"):
            value = "0.5"
        elif prefix.lower() in ("ein", "einen"):
            value = "1"
        elif prefix and (v := _de_value(prefix)) is not None:
            value = str(v)
        if value is None:
            if prefix:
                unreadable.append(m.group(0))
                return " " * len(m.group(0))
            return m.group(0)
        return f"{value} {_DE_ADJ_UNITS[m.group(2).lower()]}"

    text = re.sub(r"\b(?:a|one)\s+fortnight\b|\bfortnight\b", "2 weeks", text, flags=re.I)
    # "X and a half" before the half rule: "one and a half hours" = 1.5, "an hour and a half" = 1.5 hour
    text = re.sub(rf"\b(?:an?|one)\s+({_TIME_RE})\s+and\s+a\s+half\b", r"1.5 \1", text, flags=re.I)
    text = re.sub(r"\b(\d+)\s+and\s+a\s+half\b", lambda m: f"{int(m.group(1)) + 0.5:g}", text, flags=re.I)
    text = re.sub(rf"\b((?:{_alt(_EN_WORDS)})(?:[\s-]+(?:{_alt(_EN_WORDS)}))*)[\s-]+and[\s-]+a[\s-]+half\b",
                  lambda m: f"{_en_value(m.group(1)) + 0.5:g}" if _en_value(m.group(1)) is not None else m.group(0),
                  text, flags=re.I)
    # "tenfold", "10-fold", "zehnfach" = so many times
    text = re.sub(rf"\b({_alt(_EN_WORDS)})fold\b",
                  lambda m: f"{_en_value(m.group(1)):g} times" if _en_value(m.group(1)) is not None else m.group(0),
                  text, flags=re.I)
    text = re.sub(rf"\b({_WORD}?)fach(?:e[nmrs]?)?\b",
                  lambda m: f"{_de_value(m.group(1)):g} mal" if m.group(1) and _de_value(m.group(1)) is not None
                  else m.group(0), text, flags=re.I)
    text = re.sub(r"\b(?:(?:eine[rnm]?|1)\s+)?(drei)?viertelstunde\b",
                  lambda m: "0.75 Stunde" if m.group(1) else "0.25 Stunde", text, flags=re.I)
    text = _DE_ADJ.sub(de_adj, text)
    text = _HALF.sub("0.5 ", text)
    text = _ARTICLE_TIME.sub("1", text)

    def en(m):
        if not _FOLLOWED_BY_UNIT.match(text, m.end()):
            return m.group(0)
        first = re.split(r"[\s-]+", m.group(0).lower())[0]
        if first in _EN_SCALE and re.search(r"\d\s*$", text[:m.start()]):
            return m.group(0)  # "2 thousand euros": the magnitude pattern reads it
        v = _en_value(m.group(0))
        if v is None:
            unreadable.append(m.group(0))
            return " " * len(m.group(0))
        return f"{v:g}"

    text = _EN_RUN.sub(en, text)

    def de(m):
        w = m.group(0)
        if w.lower() in _DE_ARTICLES or not _FOLLOWED_BY_UNIT.match(text, m.end()):
            return w
        if w.lower() in ("tausend", "hundert") and re.search(r"\d\s*$", text[:m.start()]):
            return w  # "20 Tausend Euro": the magnitude pattern reads it
        v = _de_value(w)
        return w if v is None else f"{v:g}"

    return re.sub(rf"\b{_WORD}\b", de, text)


# ---------------------------------------------------------------- reading a number string
def _canon(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _readings(raw: str, lang: str | None, money: bool = False) -> frozenset[str]:
    """What a number string means, read the way `lang` writes numbers.
    en: ',' groups thousands, '.' is the decimal point. de: the reverse. Both present: the last one is decimal.
    A single separator followed by exactly 3 digits is the only ambiguous case; it is read by `lang` (both
    readings only if the language is unknown), and money never has 3 decimals. Version-like strings (8.0.1)
    are kept as written. "50,-" is 50."""
    raw = re.sub(r"[.,]-$", "", re.sub(r"[   '’]", "", raw))
    if sum(c.isdigit() for c in raw) > MAX_DIGITS:
        return frozenset({f"raw:{raw}"})
    seps = set(re.findall(r"[.,]", raw))
    groups = re.split(r"[.,]", raw)
    try:
        if not seps:
            return frozenset({_canon(Decimal(raw))})
        if len(seps) == 2:  # both: the last separator is the decimal point
            dec = raw[max(raw.rfind("."), raw.rfind(","))]
            whole, frac = raw.rsplit(dec, 1)
            return frozenset({_canon(Decimal(re.sub(r"[.,]", "", whole) + "." + frac))})
        sep = seps.pop()
        if len(groups) > 2:  # 1.000.000 / 1,000,000, or a version string
            if all(len(g) == 3 for g in groups[1:]):
                return frozenset({_canon(Decimal("".join(groups)))})
            return frozenset({f"raw:{raw}"})
        whole, frac = groups
        thousands = _canon(Decimal(whole + frac))
        decimal = _canon(Decimal(f"{whole}.{frac}"))
        if len(frac) != 3:
            return frozenset({decimal})
        if money:  # no currency has 3 decimals: "€1.000" and "€2,500" are thousands in any language
            return frozenset({thousands})
        if lang == "en":
            return frozenset({thousands if sep == "," else decimal})
        if lang == "de":
            return frozenset({thousands if sep == "." else decimal})
        return frozenset({thousands, decimal})
    except (InvalidOperation, ValueError):
        return frozenset({f"raw:{raw}"})


# ---------------------------------------------------------------- dates and times
_MONTHS = {}
for _i, _names in enumerate(["january jan januar jänner jaenner", "february feb februar", "march mar märz maerz mrz",
                             "april apr", "may mai", "june jun juni", "july jul juli", "august aug",
                             "september sep sept", "october oct oktober okt", "november nov", "december dec dezember dez"], 1):
    for _n in _names.split():
        _MONTHS[_n] = _i
_MONTH_RE = _alt(_MONTHS)


def _iso(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    return f"{y:04d}-{m:02d}-{d:02d}" if 1 <= m <= 12 and 1 <= d <= 31 else None


def _date_readings(s: str, lang: str | None) -> frozenset[str]:
    """Dates as 'date:YYYY-MM-DD' (or 'md:MM-DD' without a year). A slash date whose day and month could be
    swapped is only matched exactly ('slash:...') unless the ticket is German (always DD/MM)."""
    s = s.strip()
    out: set = set()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        out.add(_iso(int(m[1]), int(m[2]), int(m[3])))
    elif m := re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", s):
        out.add(_iso(int(m[3]), int(m[2]), int(m[1])))
    elif m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if lang == "de" or a > 12:
            out.add(_iso(y, b, a))
        elif b > 12:
            out.add(_iso(y, a, b))
        else:
            return frozenset({f"slash:{a}/{b}/{y % 100}"})
    out.discard(None)
    return frozenset({f"date:{x}" for x in out} or {f"raw:{s}"})


def _month_date(m) -> frozenset[str]:
    g = m.groupdict()
    day, month = int(g.get("d1") or g.get("d2")), _MONTHS[(g.get("m1") or g.get("m2")).lower().rstrip(".")]
    year = g.get("y1") or g.get("y2") or g.get("y3")
    if year:
        iso = _iso(int(year), month, day)
        return frozenset({f"date:{iso}"} if iso else {f"raw:{m.group(0)}"})
    return frozenset({f"md:{month:02d}-{day:02d}"})


def _time_value(h: int, mins: str, ampm: str | None) -> frozenset[str]:
    ampm = (ampm or "").lower().replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return frozenset({f"time:{h}:{mins or '00'}"})


# ---------------------------------------------------------------- extraction
_RATE_RE = (rf"\s*(?:/|per|pro|je)\s*(?:1\s+)?(?P<rw>{_WORD})\b|\s+1\s+(?P<rw2>{_TIME_RE})\b"
            rf"|\s+(?P<adv>{_alt(_RATE_ADV)})")


def _scale(vals: frozenset, mag: str | None) -> frozenset:
    """Apply a magnitude ("k", "Mio.", " million") to every numeric reading."""
    if not mag:
        return vals
    f = _MAGNITUDE.get(mag.strip().lower().rstrip("."))
    if not f:
        return vals
    out = set()
    for v in vals:
        try:
            out.add(v if v.startswith("raw:") else _canon(Decimal(v) * f))
        except InvalidOperation:
            out.add(v)
    return frozenset(out)


def _rate(m) -> str:
    """The billing period after an amount, as a canonical unit ('' if none)."""
    if not m.group("rate"):
        return ""
    if m.group("adv"):
        return "/" + _RATE_ADV[m.group("adv").lower()]
    w = (m.group("rw") or m.group("rw2")).lower()
    return "/" + _UNITS.get(w, w)


_PATTERNS = [  # most specific first; each match is blanked so later patterns can't re-read it
    ("mdate", re.compile(rf"\b(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\.?\s+(?P<m1>{_MONTH_RE})\b\.?(?:\s*,?\s*(?P<y1>\d{{4}}))?"
                         rf"|\b(?P<m2>{_MONTH_RE})\b\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(?P<y2>\d{{4}}))?", re.I)),
    ("date", re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("time", re.compile(r"\b(?P<h>\d{1,2})(?::(?P<mm>\d{2}))?\s?(?P<ap>a\.?m\.?|p\.?m\.?)(?![a-z])"
                        r"|\b(?P<h2>\d{1,2})(?::(?P<mm2>\d{2}))?\s?Uhr\b|\b(?P<h3>\d{1,2}):(?P<mm3>\d{2})\b", re.I)),
    ("always", re.compile(r"\b24\s?/\s?7\b")),
    ("range", re.compile(rf"(?P<a>{_NUM})\s*(?:-|–|to|bis|until)\s*(?P<b>{_NUM}){_GAP_ANY}"
                         rf"(?P<u>{_UNIT_RE}|{_PCT_RE}|{_CUR_WORD_RE}|{_CUR_SYM_RE})(?![a-zäöüß])", re.I)),
    ("cur", re.compile(rf"(?:(?P<c1>{_CUR_SYM_RE}|\b(?:{_CUR_WORD_RE})\b)\s?(?P<n1>{_NUM})(?P<g1>{_MAG_CUR_RE})?"
                       rf"|(?P<n2>{_NUM})(?P<g2>{_MAG_CUR_RE})?{_GAP_ANY}(?P<c2>{_CUR_SYM_RE}|\b(?:{_CUR_WORD_RE})\b))"
                       rf"(?P<rate>{_RATE_RE})?", re.I)),
    ("pct", re.compile(rf"(?P<n>{_NUM}){_GAP}(?:{_PCT_RE})(?![a-z])(?P<rate>{_RATE_RE})?", re.I)),
    ("qty", re.compile(rf"(?P<n>{_NUM})(?P<g>{_MAG_WORD_RE})?{_GAP_ANY}(?P<u>{_UNIT_RE})\b", re.I)),
    ("qtyde", re.compile(rf"(?P<n>{_NUM})[\s-]?(?P<w>{_DE_COMPOUND_UNIT})\b", re.I)),  # Geschäftstagen
    ("mult", _MULTIPLE),
    ("frac", _FRACTION_RE),
    ("num", re.compile(rf"(?P<n>{_NUM})(?P<g>{_MAG_WORD_RE})?", re.I)),
]
TYPED = ("cur", "pct", "qty", "frac", "amount")
# a bare number right after a money word is an amount ("a refund of 500"): it needs a retrieved answer
_MONEY_CUE = re.compile(r"(?:refund|credit|charg|compensat|waiv|reimburs|discount|fee|cost|price|pay|bonus|voucher|"
                        r"gutschein|erstatt|gutschrift|gutschreib|rabatt|gebühr|kosten|preis|zahl|entschädig|vergüt)"
                        r"\w*\W+(?:\w+\W+){0,3}$", re.I)
_ID_BEFORE = re.compile(r"(?:order|bestellung|bestellnummer|invoice|rechnung|ticket|case|fall|nr|no|number|nummer|"
                        r"ref|reference|referenz|id|version|ver|v|error|fehler|code|step|schritt|page|seite|line|"
                        r"zeile|model|modell|serial|seriennummer|kunde|customer|account|konto)\.?\s*[#:]?\s*$", re.I)
_DE_COMPOUND_CANON = (("tag", "day"), ("stunde", "hour"), ("woche", "week"), ("monat", "month"), ("jahr", "year"))
_PRODUCT_BEFORE = re.compile(r"([A-Za-z][\w+]*)\s+$")


def _typed(unit_token: str, number: str, lang: str | None) -> tuple[str, frozenset]:
    u = unit_token.lower()
    if u in _CURRENCY:
        return "cur", frozenset(f"{_CURRENCY[u]}:{r}" for r in _readings(number, lang, money=True))
    if u in _PCT_WORDS or u == "%":
        return "pct", frozenset(f"%:{r}" for r in _readings(number, lang))
    return "qty", frozenset(f"{_UNITS[u]}:{r}" for r in _readings(number, lang))


def extract_values(text: str, lang: str | None = None, context: str | None = None) -> list[tuple[str, frozenset, str]]:
    """Every number-bearing claim as (kind, possible normalised values, the text as written).
    A quantity written in words that can't be read comes back with no values, so it never matches.
    `context` (ticket + sources, lowercased) lets "Windows 10 users" read as a version when it names a
    product they mention; without it (reading a source) such a number is read both ways."""
    text = _PH_ANY.sub(" ", _TKT.sub(" ", text or ""))
    unreadable: list[str] = []
    text = _convert_number_words(text, unreadable)
    found = [("unreadable", frozenset(), s.strip()) for s in unreadable]
    for kind, rx in _PATTERNS:
        def take(m, kind=kind, src=text):
            s = m.group(0)
            if kind == "mdate":
                found.append(("date", _month_date(m), s.strip()))
            elif kind == "date":
                found.append(("date", _date_readings(s, lang), s.strip()))
            elif kind == "time":
                g = m.groupdict()
                h, mins, ap = ((g["h"], g["mm"], g["ap"]) if g["h"] else (g["h2"], g["mm2"], None) if g["h2"]
                               else (g["h3"], g["mm3"], None))
                found.append(("time", _time_value(int(h), mins, ap), s.strip()))
            elif kind == "always":
                found.append(("always", frozenset({"always:24/7"}), s.strip()))
            elif kind == "range":  # "1 to 5 business days": both ends carry the unit
                for n in (m.group("a"), m.group("b")):
                    k, vals = _typed(m.group("u"), n, lang)
                    found.append((k, vals, f"{n} {m.group('u')} (in '{s.strip()}')"))
            elif kind == "cur":
                n, c, g = ((m.group("n1"), m.group("c1"), m.group("g1")) if m.group("n1")
                           else (m.group("n2"), m.group("c2"), m.group("g2")))
                per, cur = _rate(m), _CURRENCY[c.lower()]
                nums = _scale(_readings(n, lang, money=True), g)
                found.append(("cur", frozenset(f"{cur}:{r}{per}" for r in nums), s.strip()))
            elif kind == "pct":
                per = _rate(m)
                found.append(("pct", frozenset(f"%:{r}{per}" for r in _readings(m.group("n"), lang)), s.strip()))
            elif kind == "qtyde":
                w = m.group("w").lower()
                unit = next(c for stem, c in _DE_COMPOUND_CANON if stem in w)
                found.append(("qty", frozenset(f"{unit}:{r}" for r in _readings(m.group("n"), lang)), s.strip()))
            elif kind == "qty":
                n, u = m.group("n"), m.group("u")
                vals = frozenset(f"{_UNITS[u.lower()]}:{r}" for r in _scale(_readings(n, lang), m.group("g")))
                prod = _PRODUCT_BEFORE.search(src, 0, m.start())
                if prod and any(c.isupper() or c.isdigit() for c in prod.group(1)):  # Windows, iOS, PHP, M365
                    named = f"{prod.group(1)} {n}".lower()
                    if context is None:  # a source: "Windows 10 users" also grounds a plain 10
                        found.append(("num", _readings(n, lang), f"{prod.group(1)} {n}"))
                    elif named in context:  # the reply names a product version the ticket/sources mention
                        found.append(("num", _readings(n, lang), f"{prod.group(1)} {n}"))
                        return " " * len(s)
                found.append(("qty", vals, s.strip()))
            elif kind == "mult":
                v = "3" if re.match(r"(?:triple|dreifach)", s, re.I) else "2"
                found.append(("frac", frozenset({f"times:{v}"}), s.strip()))
            elif kind == "frac":
                found.append(("frac", frozenset({f"frac:{_FRACTION[s.lower()]}"}), s.strip()))
            else:
                vals = _scale(_readings(m.group("n"), lang), m.group("g"))
                if (context is not None and _MONEY_CUE.search(src, max(0, m.start() - 60), m.start())
                        and not _ID_BEFORE.search(src, max(0, m.start() - 30), m.start())):
                    found.append(("amount", frozenset(f"any:{v}" for v in vals), s.strip()))
                else:
                    found.append(("num", vals, s.strip()))
            return " " * len(s)
        text = rx.sub(take, text)
    return found


# ---------------------------------------------------------------- the checks
def _check_format(draft):
    if not isinstance(draft, dict):
        return False, f"draft is {type(draft).__name__}, not a dict", "", [], []
    reply, cited, asks = draft.get("reply"), draft.get("cited_ids"), draft.get("unsupported_asks")
    problems = []
    if not isinstance(reply, str) or not reply.strip():
        problems.append("reply is missing or empty")
        reply = reply if isinstance(reply, str) else ""
    else:
        if len(reply.split()) > MAX_WORDS:
            problems.append(f"reply has {len(reply.split())} words (max {MAX_WORDS})")
        if len(reply) > MAX_CHARS:
            problems.append(f"reply has {len(reply)} characters (max {MAX_CHARS})")
    if not isinstance(cited, list):
        problems.append("cited_ids is not a list")
        cited = []
    if not isinstance(asks, list):
        problems.append("unsupported_asks is not a list")
        asks = []
    ev = "; ".join(problems) or f"{len(reply.split())} words, fields well-formed"
    return not problems, ev, reply[:MAX_CHARS], cited, asks  # later checks never see more than MAX_CHARS


def _check_citations(reply, cited, retrieved_ids):
    ids = {c.strip() for c in cited if isinstance(c, str)} | set(_TKT.findall(reply))
    if any(not isinstance(c, str) for c in cited):
        return False, f"cited_ids contains non-text entries: {[c for c in cited if not isinstance(c, str)][:3]}"
    bad, good = sorted(ids - retrieved_ids), sorted(ids & retrieved_ids)
    if bad:
        return False, f"cites {bad}, not among the retrieved tickets"
    if not good:
        return False, "no ticket is cited"
    return True, f"cites {good}, all retrieved"


def _check_numbers(reply, lang, answers, ticket_text):
    """Amounts, percentages and quantities with a unit must be in a retrieved answer with the same unit.
    Plain numbers, dates and times may also come from the customer's ticket (versions, order numbers)."""
    context = " ".join([ticket_text] + [t for t, _ in answers]).lower()
    claims = extract_values(reply, lang, context=context)
    if not claims:
        return True, "no numbers in the reply"

    def grounding(vals):  # what a source value also vouches for
        out = set(vals)
        for v in vals:
            if v.startswith("date:"):  # a full date also grounds the same day without a year
                out.add("md:" + v[10:])
            if v.startswith("%:") and v[2:] in ("50", "25"):  # 50% also grounds "half", 25% "a quarter"
                out.add("frac:" + {"50": "0.5", "25": "0.25"}[v[2:]])
            if ":" in v and not v.startswith(("raw:", "date:", "md:", "time:", "slash:", "always:")):
                base = v.split("/", 1)[0]  # "EUR:20/year" also grounds "EUR:20"
                out.add(base)
                out.add("any:" + base.split(":", 1)[1])  # and a bare amount "20" after a money word
        return out

    typed_ok, plain_ok = set(), set()
    for text, src_lang in answers:
        for kind, vals, _ in extract_values(text, src_lang):
            (typed_ok if kind in TYPED else plain_ok).update(grounding(vals))
    for kind, vals, _ in extract_values(ticket_text, lang):
        if kind not in TYPED:
            plain_ok.update(grounding(vals))
    missing = [s for kind, vals, s in claims if not (vals & (typed_ok if kind in TYPED else plain_ok))]
    if missing:
        return False, f"not grounded (amounts, % and durations need a retrieved answer): {missing}"
    return True, f"all {len(claims)} grounded: {[s for _, _, s in claims][:6]}"


def _norm_link(u: str) -> str:
    u = u.strip().rstrip(".,;:!?)").lower()
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)
    return re.sub(r"^www\.", "", u).rstrip("/")


def _links(text: str) -> set[str]:
    """Normalised links in a text: URLs plus bare host.tld[/path], minus file names and Sentence.Joins."""
    out = {_norm_link(u) for u in _URL.findall(text)}
    for d in _DOMAIN.findall(text):
        host = d.split("/", 1)[0]
        tld = host.rsplit(".", 1)[-1]
        if tld.lower() in _FILE_EXT or tld != tld.lower():
            continue
        out.add(_norm_link(d))
    out.discard("")
    return out


def _check_placeholders(reply, sources):
    allowed = {p.lower() for s in sources for p in _PH_ANY.findall(s)}
    unknown = sorted({p for p in _PH_ANY.findall(reply) if p.lower() not in allowed})
    problems = []
    if unknown:
        problems.append(f"placeholders not in any source: {unknown}")

    # Personal data the model wrote itself: mask the reply (existing placeholders removed first, so a new
    # <email> can't hide behind an old one) and list what got replaced, unless it is quoted from a source.
    plain = _PH_ANY.sub(" ", reply)
    masked = mask(plain)
    squash = lambda s: re.sub(r"\s+", " ", s).strip()
    corpus = squash(" ".join(sources))
    leaked = []
    if masked != plain:  # word-level diff: fast, and the evidence is whole words
        pw, mw = plain.split(), masked.split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pw, mw, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            span = " ".join(pw[i1:i2]).strip(".,;:!?()")
            tags = sorted(set(_PH_ANY.findall(" ".join(mw[j1:j2])))) or ["<pii>"]
            if span and span not in corpus and not any(span in x for x in leaked):
                leaked.append(f"{span!r} -> {'/'.join(tags)}")
    known = set()
    for src in sources:
        for link in _links(src):
            known.add(link)
            known.add(link.split("/", 1)[0])  # a source's full link also allows its bare host
    leaked += [f"{u!r} (link)" for u in sorted(_links(reply)) if u not in known]
    leaked += [f"{e!r} (email)" for e in _OBF_EMAIL.findall(reply)]
    if leaked:
        problems.append(f"personal data or links not in any source (invented by the model): {leaked[:4]}")
    if problems:
        return False, "; ".join(problems)
    return True, f"placeholders kept: {sorted(set(_PH_ANY.findall(reply))) or 'none used'}"


def _check_language(reply, lang):
    if lang not in ("en", "de"):
        return False, f"ticket language {lang!r} is not en/de"
    clean = _URL.sub(" ", _PH_ANY.sub(" ", _TKT.sub(" ", reply)))
    n_words = len(_ALPHA_WORD.findall(clean))
    if n_words < MIN_WORDS_FOR_LANG:
        return True, f"too short to judge ({n_words} words); not enforced"
    whole = detect_language_with_confidence(body=clean)
    if whole.language != lang:
        return False, f"reply detected as '{whole.language}' ({whole.confidence:.2f}), ticket is '{lang}'"
    # every full sentence, and every window of 2-3 consecutive short sentences, must be in `lang`
    sents = [x.strip() for x in re.split(r"[.!?\n;:]+", clean) if _ALPHA_WORD.search(x)]
    off = []
    for i in range(len(sents)):
        for j in range(i + 1, min(i + 4, len(sents) + 1)):
            chunk = " ".join(sents[i:j])
            if len(_ALPHA_WORD.findall(chunk)) >= MIN_WORDS_PER_SENTENCE:
                p = detect_language_with_confidence(body=chunk)
                if p.language != lang and p.confidence >= SENTENCE_LANG_CONF and chunk[:60] not in off:
                    off.append(chunk[:60])
                break
    if off:
        return False, f"sentences not in '{lang}': {off[:2]}"
    return True, f"reply and all full sentences are '{lang}' ({whole.confidence:.2f})"


def _check_unsupported(asks):
    asks = [str(a) for a in asks if str(a).strip()]
    if asks:
        return False, f"model could not ground: {asks[:3]}"
    return True, "none"


# ---------------------------------------------------------------- public entry point
_INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff\u00ad]")


def _clean(text: str, cap: int) -> str:
    """NFKC (full-width / superscript digits -> ASCII, compatibility symbols), drop zero-width characters."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text[:cap]))


def verify(draft: dict, retrieved: list[dict], ticket_text: str, lang: str) -> dict:
    """Run every check. passed is True only if all pass; failures carry the code the gate escalates with."""
    rows = [r for r in (retrieved if isinstance(retrieved, list) else [])
            if isinstance(r, dict) and isinstance(r.get("ticket_id"), str) and r["ticket_id"].strip()]
    retrieved_ids = {r["ticket_id"].strip() for r in rows}
    lang = str(lang or "").strip().lower()
    ticket_text = _clean("" if ticket_text is None else str(ticket_text), MAX_SOURCE_CHARS)
    answers = []
    for r in rows:
        text = normalize_placeholders(_clean(str(r.get("answer") or ""), MAX_SOURCE_CHARS))
        # language from the text, never the row's language column (wrong on ~9% of rows, see language.py)
        src_lang = detect_language_with_confidence(body=text).language if text.strip() else None
        answers.append((text, src_lang))
    sources = [a for a, _ in answers] + [ticket_text]

    fmt_ok, fmt_ev, reply, cited, asks = _check_format(draft)
    reply = _clean(reply, MAX_CHARS)
    # numbers are grounded in the answers the draft cites (all retrieved ones if it cites none valid)
    cited_ids = {c.strip() for c in cited if isinstance(c, str)} | set(_TKT.findall(reply))
    cited_answers = [a for r, a in zip(rows, answers) if r["ticket_id"].strip() in cited_ids] or answers
    results = [
        ("format", VERIFY_FAILED, fmt_ok, fmt_ev),
        ("citations", VERIFY_FAILED, *_check_citations(reply, cited, retrieved_ids)),
        ("numbers", VERIFY_FAILED, *_check_numbers(reply, lang if lang in ("en", "de") else None, cited_answers, ticket_text)),
        ("placeholders", VERIFY_FAILED, *_check_placeholders(reply, sources)),
        ("language", VERIFY_FAILED, *_check_language(reply, lang)),
        ("unsupported_asks", UNSUPPORTED_ASK, *_check_unsupported(asks)),
    ]
    checks = [{"check": n, "passed": ok, "evidence": ev} for n, _, ok, ev in results]
    failures = [{"check": n, "code": code, "detail": ev} for n, code, ok, ev in results if not ok]
    return {"passed": not failures, "failures": failures, "checks": checks}
