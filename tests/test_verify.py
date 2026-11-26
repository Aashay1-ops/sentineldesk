"""Tests for ⑤ VERIFY (sentineldesk/verify.py). Run: python -m pytest tests/test_verify.py -q"""
from __future__ import annotations

import time

import pytest

from sentineldesk.verify import UNSUPPORTED_ASK, VERIFY_FAILED, extract_values, verify

RETRIEVED = [
    {"ticket_id": "TKT-17145", "answer": "Please restart the VPN client and clear the cache. "
                                         "If it persists, call <tel_num> and quote <acc_num>."},
    {"ticket_id": "TKT-03282", "answer": "Refunds for double charges are processed within 24 hours. "
                                         "The fee of 1000.50 EUR will be credited, a 10% discount applies."},
    {"ticket_id": "TKT-05470", "answer": "Sehr geehrte/r <name>, wir prüfen die Abbuchung innerhalb von 3 Tagen."},
]
TICKET_EN = "VPN down\nOur VPN client fails since this morning on PHP 8.0 servers."
TICKET_DE = "Doppelte Abbuchung\nMir wurde die Gebühr zweimal abgebucht. Bitte prüfen."


def draft(reply, cited=("TKT-17145",), asks=()):
    return {"reply": reply, "cited_ids": list(cited), "unsupported_asks": list(asks)}


def failed(result):
    return {f["check"] for f in result["failures"]}


# ---------------------------------------------------------------- happy path
def test_clean_reply_passes():
    r = verify(draft("Thanks for reaching out. Please restart the VPN client and clear the cache [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]
    assert [c["check"] for c in r["checks"]] == ["format", "citations", "numbers", "placeholders",
                                                  "language", "unsupported_asks"]


# ---------------------------------------------------------------- citations
def test_unretrieved_citation_fails():
    r = verify(draft("Please restart the VPN client and clear the cache, see the earlier case.",
                     cited=["TKT-99999"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"citations"}


def test_inline_unretrieved_citation_fails_even_if_field_is_clean():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145] [TKT-12345]."),
               RETRIEVED, TICKET_EN, "en")
    assert "citations" in failed(r)
    assert "TKT-12345" in r["failures"][0]["detail"]


def test_no_citation_fails():
    r = verify(draft("Please restart the VPN client and clear the cache today.", cited=[]),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"citations"}


# ---------------------------------------------------------------- numbers
def test_invented_amount_fails():
    r = verify(draft("We will credit you €50 for the trouble with the VPN client [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}
    assert "€50" in r["failures"][0]["detail"]


def test_invented_duration_fails():
    r = verify(draft("Your refund for the double charge will arrive within 48 hours [TKT-03282].",
                     cited=["TKT-03282"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}


def test_grounded_duration_passes_across_languages():
    # "24 hours" in the reply, "24 hours" in the source; and the German "3 Tagen" matches "3 days".
    r = verify(draft("Refunds for double charges are processed within 24 hours, review takes 3 days "
                     "[TKT-03282][TKT-05470].", cited=["TKT-03282", "TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


def test_number_word_before_unit_is_checked():
    r = verify(draft("The review of your double charge takes three days in total [TKT-05470].",
                     cited=["TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]  # "three days" == "3 Tagen"
    r = verify(draft("The review of your double charge takes five days in total [TKT-05470].",
                     cited=["TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}


def test_plain_number_word_is_not_a_claim():
    assert extract_values("One of our agents will help you.") == []


def test_german_number_format_matches_english_source():
    r = verify(draft("Die Gebühr von 1.000,50 € wird gutgeschrieben, es gilt ein Rabatt von 10 % [TKT-03282].",
                     cited=["TKT-03282"]), RETRIEVED, TICKET_DE, "de")
    assert "numbers" not in failed(r), r["failures"]


def test_currency_must_match_not_just_the_number():
    r = verify(draft("The fee of 1000.50 USD will be credited back to you [TKT-03282].", cited=["TKT-03282"]),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}  # the source says EUR


def test_number_from_customer_ticket_is_allowed():
    r = verify(draft("Please restart the VPN client on your PHP 8.0 servers and clear the cache [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


# ---------------------------------------------------------------- placeholders and invented PII
def test_kept_placeholders_pass():
    r = verify(draft("If the VPN client still fails, call <tel_num> and quote <acc_num> [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


def test_invented_phone_number_fails():
    r = verify(draft("If the VPN client still fails, call us on +49 30 1234567 any time [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


def test_invented_email_fails():
    r = verify(draft("If the VPN client still fails, write to help@example.com for support [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


def test_unknown_placeholder_fails():
    r = verify(draft("If the VPN client still fails, contact <manager_name> directly please [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


# ---------------------------------------------------------------- language
def test_wrong_language_fails():
    r = verify(draft("Thank you, please restart the VPN client and clear the cache today [TKT-17145]."),
               RETRIEVED, TICKET_DE, "de")
    assert "language" in failed(r)


def test_short_reply_language_not_enforced():
    r = verify(draft("Please restart it [TKT-17145]."), RETRIEVED, TICKET_DE, "de")
    assert "language" not in failed(r)


def test_german_reply_to_german_ticket_passes():
    r = verify(draft("Sehr geehrte/r <name>, wir prüfen die Abbuchung innerhalb von 3 Tagen und melden uns "
                     "bei Ihnen [TKT-05470].", cited=["TKT-05470"]), RETRIEVED, TICKET_DE, "de")
    assert r["passed"], r["failures"]


# ---------------------------------------------------------------- unsupported asks
def test_unsupported_ask_fails_with_its_own_code():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145].",
                     asks=["refund guarantee for downtime"]), RETRIEVED, TICKET_EN, "en")
    assert r["failures"] == [{"check": "unsupported_asks", "code": UNSUPPORTED_ASK,
                              "detail": "model could not ground: ['refund guarantee for downtime']"}]


# ---------------------------------------------------------------- malformed input never crashes
@pytest.mark.parametrize("bad", [None, "just a string", {}, {"reply": ""}, {"reply": 42, "cited_ids": "TKT-17145"}])
def test_malformed_draft_fails_without_crashing(bad):
    r = verify(bad, RETRIEVED, TICKET_EN, "en")
    assert not r["passed"]
    assert "format" in failed(r)
    assert all(f["code"] == VERIFY_FAILED for f in r["failures"] if f["check"] == "format")


def test_empty_retrieval_fails_citations():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145]."), [], TICKET_EN, "en")
    assert "citations" in failed(r)


def test_too_long_reply_fails():
    r = verify(draft("word " * 300 + "[TKT-17145]"), RETRIEVED, TICKET_EN, "en")
    assert "format" in failed(r)


def test_all_failures_are_reported_not_just_the_first():
    r = verify(draft("Call +49 30 1234567, we credit €50 within 48 hours.", cited=["TKT-99999"],
                     asks=["compensation"]), RETRIEVED, TICKET_EN, "en")
    assert {"citations", "numbers", "placeholders", "unsupported_asks"} <= failed(r)


def test_fast_enough_for_the_live_path():
    d = draft("Refunds for double charges are processed within 24 hours [TKT-03282].", cited=["TKT-03282"])
    t = time.perf_counter()
    for _ in range(200):
        verify(d, RETRIEVED, TICKET_EN, "en")
    assert (time.perf_counter() - t) / 200 < 0.01  # under 10 ms per draft


# ================================================================ regressions from the cold review of PR #5
def one(answer, language=None, tid="TKT-00001"):
    row = {"ticket_id": tid, "answer": answer}
    if language:
        row["language"] = language
    return [row]


def check(reply, answer, lang="en", ticket="Please help with my account.", language=None):
    return verify({"reply": reply, "cited_ids": ["TKT-00001"], "unsupported_asks": []},
                  one(answer, language), ticket, lang)


# 1. units must match: a number with an unknown or different unit can't borrow a bare number from a source
@pytest.mark.parametrize("reply,answer,lang", [
    ("Your device comes with a 3 years warranty from us [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.", "en"),
    ("Your request is processed in 10 seconds by our team [TKT-00001].", "A 10% discount applies to renewals.", "en"),
    ("Your plan is guaranteed for 24 more months from today [TKT-00001].", "Requests are handled within 24 hours.", "en"),
    ("We answer within 10 business hours on weekdays [TKT-00001].", "We answer within 10 days on weekdays.", "en"),
    ("Sie erhalten 10 Jahre lang einen Rabatt auf alles [TKT-00001].", "Es gilt ein Rabatt von 10 %.", "de"),
    ("You will be credited 10 times the monthly fee [TKT-00001].", "A 10% discount applies to renewals.", "en"),
])
def test_unit_mismatch_fails(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_filler_word_between_number_and_unit_still_matches():
    r = check("We review your case within 3 business days at most [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.")
    assert "numbers" not in failed(r), r["failures"]


# 2. number words: compounds read correctly, anything unreadable fails
@pytest.mark.parametrize("reply,answer,lang", [
    ("Your refund arrives within fifteen days from today [TKT-00001].", "Refunds take 3 days.", "en"),
    ("You get a fifty percent discount on the plan [TKT-00001].", "A 10% discount applies.", "en"),
    ("Sie erhalten zwanzig Euro als Gutschrift von uns [TKT-00001].", "Sie erhalten 10 Euro Gutschrift.", "de"),
    ("We will credit you a hundred euros for this [TKT-00001].", "We credit 10 euros.", "en"),
    ("Die Prüfung dauert eine Woche bei uns im Team [TKT-00001].", "Die Prüfung dauert 3 Tage.", "de"),
    ("Your refund arrives within thirty-three days from today [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.", "en"),
])
def test_invented_number_words_fail(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


@pytest.mark.parametrize("reply,answer,lang", [
    ("Requests are handled within twenty-four hours by our team [TKT-00001].", "Requests are handled within 24 hours.", "en"),
    ("Anfragen werden innerhalb von vierundzwanzig Stunden bearbeitet [TKT-00001].", "Requests take 24 hours.", "de"),
])
def test_compound_number_words_match(reply, answer, lang):
    r = check(reply, answer, lang)
    assert "numbers" not in failed(r), r["failures"]


# 3. a separator is read by language: no 1000x amounts
@pytest.mark.parametrize("reply,answer,lang", [
    ("The plan costs €2,500 per user per month for you [TKT-00001].", "The plan costs €2.50 per user.", "en"),
    ("We add a credit of €1,000 to your account today [TKT-00001].", "A credit of €1 is added.", "en"),
    ("Wir schreiben Ihnen 1.000 € auf Ihrem Konto gut [TKT-00001].", "A credit of €1 is added.", "de"),
])
def test_amount_scale_mismatch_fails(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_same_amount_in_both_formats_matches():
    r = check("Wir schreiben Ihnen 1.000 € auf Ihrem Konto gut [TKT-00001].", "A credit of €1,000 is added.", "de",
              language="en")
    assert "numbers" not in failed(r), r["failures"]


# 4. an invented email can't hide behind an existing <email> placeholder
def test_invented_email_next_to_placeholder_fails():
    r = check("Please write to <email> or directly to billing@contoso.com for help [TKT-00001].",
              "Please write to <email> for help.")
    assert "placeholders" in failed(r)


# 5. huge numbers never crash
@pytest.mark.parametrize("reply", ["€" + "1" * 400 + ",5 credit for you today [TKT-00001].",
                                   "Reference " + "9" * 4400 + " for you [TKT-00001]."])
def test_huge_numbers_fail_without_crashing(reply):
    r = check(reply, "A credit of €1 is added.")
    assert not r["passed"]


# 6. a German greeting can't carry an English reply past the language check
def test_mixed_language_reply_fails():
    r = check("Sehr geehrte/r <name>, vielen Dank für Ihre Nachricht. Please restart the VPN client and clear the "
              "cache, then try again [TKT-00001].", "Sehr geehrte/r <name>, bitte VPN neu starten.", "de")
    assert "language" in failed(r)


# 7. figures quoted verbatim from a source are not flagged as invented phone numbers
@pytest.mark.parametrize("reply,answer,lang", [
    ("Das Limit für Ihr Konto liegt bei 1.000.000 EUR pro Jahr [TKT-00001].",
     "Das Limit für Ihr Konto liegt bei 1.000.000 EUR pro Jahr.", "de"),
    ("Die Wartung beginnt am 21.11.2026 08:00 und dauert kurz [TKT-00001].",
     "Die Wartung beginnt am 21.11.2026 08:00.", "de"),
])
def test_grounded_figures_are_not_pii(reply, answer, lang):
    r = check(reply, answer, lang)
    assert "placeholders" not in failed(r), r["failures"]
    assert "numbers" not in failed(r), r["failures"]


# 8. odd inputs never crash; rows without a ticket_id can't be cited
def test_bad_retrieved_rows_and_ticket_types_do_not_crash():
    rows = [None, "x", {"answer": "no id here"}, {"ticket_id": "TKT-00001", "answer": "Restart the VPN client."}]
    r = verify({"reply": "Please restart the VPN client now [TKT-00001].", "cited_ids": ["TKT-00001"],
                "unsupported_asks": []}, rows, 12345, "en")
    assert r["passed"], r["failures"]


def test_row_without_ticket_id_cannot_be_cited():
    r = verify({"reply": "Please restart the VPN client now.", "cited_ids": ["None"], "unsupported_asks": []},
               [{"answer": "Restart the VPN client."}], "ticket", "en")
    assert "citations" in failed(r)


def test_uppercase_lang_is_accepted():
    r = check("Please restart the VPN client and clear the cache today [TKT-00001].", "Restart the VPN client.", "EN")
    assert "language" not in failed(r), r["failures"]


# 9. the same date in another format matches
def test_date_formats_match_across_languages():
    r = check("Die Wartung findet am 21.11.2026 um 10:30 statt, danach geht es weiter [TKT-00001].",
              "Maintenance is on 2026-11-21 at 10:30.", "de", language="en")
    assert "numbers" not in failed(r), r["failures"]


# 10. a huge one-word reply fails fast
def test_huge_single_token_reply_is_fast():
    t = time.perf_counter()
    r = check("1-" * 20000 + " [TKT-00001]", "Restart the VPN client.")
    assert "format" in failed(r)
    assert time.perf_counter() - t < 1.0


# 11. uppercase placeholders, invented links and spelled-out emails are caught; quoted links are fine
@pytest.mark.parametrize("reply", [
    "Please contact <Manager> about the VPN client issue [TKT-00001].",
    "Please download the fix from https://contoso-fix.example/vpn today [TKT-00001].",
    "Please write to help (at) contoso (dot) com about this [TKT-00001].",
])
def test_invented_placeholders_links_emails_fail(reply):
    assert "placeholders" in failed(check(reply, "Restart the VPN client."))


def test_link_quoted_from_source_passes():
    r = check("The guide is at https://docs.example.com/vpn for you [TKT-00001].",
              "See the guide at https://docs.example.com/vpn for steps.")
    assert "placeholders" not in failed(r), r["failures"]


# 12. the customer's own amount is not enough to promise it; plain numbers from the ticket still are
def test_amount_from_ticket_alone_cannot_be_promised():
    r = check("We will refund the €500 to your account shortly [TKT-00001].", "We review refund requests.",
              ticket="I was charged €500 twice, please refund.")
    assert "numbers" in failed(r)


# second pass, from re-running the reviewer's probes
@pytest.mark.parametrize("reply", ["A credit of €1.000 has been issued to your account today [TKT-00001].",
                                   "A credit of 1 000 € has been issued to your account today [TKT-00001]."])
def test_three_decimal_amounts_are_thousands(reply):
    assert "numbers" in failed(check(reply, "A credit of €1 is added."))


def test_space_separated_thousands_match():
    r = check("A credit of 1 000 € has been issued to your account today [TKT-00001].",
              "A credit of €1,000 is added.")
    assert "numbers" not in failed(r), r["failures"]


def test_leak_evidence_shows_the_whole_address():
    r = check("Please write to <email> or directly to billing@contoso.com for help [TKT-00001].",
              "Please write to <email> for help.")
    assert "billing@contoso.com" in next(f["detail"] for f in r["failures"] if f["check"] == "placeholders")


# ================================================================ regressions from the second cold review
A5 = "Refunds are processed within 5 business days."


# 1-3 + extra: durations/amounts that carry no digit, sit away from their unit, use unusual forms, or come as ranges
@pytest.mark.parametrize("reply,answer,lang,ticket", [
    ("Refunds are processed within an hour after we confirm the return [TKT-00001].", A5, "en", "Refund please."),
    ("Refunds are processed within a day after we confirm the return [TKT-00001].", A5, "en", "Refund please."),
    ("You will receive half of the amount back within 5 business days [TKT-00001].", A5, "en", "Refund please."),
    ("Die Rückerstattung erfolgt innerhalb einer halben Stunde auf Ihr Konto [TKT-00001].",
     "Rückerstattungen dauern 5 Werktage.", "de", "Erstattung bitte."),
    ("As an apology we will add two free months to your subscription today [TKT-00001].", A5, "en", "Refund please."),
    ("As an apology we will add 2 free months to your subscription today [TKT-00001].", A5, "en",
     "I ordered 3 weeks ago, did step 1 and 2."),
    ("Als Entschädigung erstatten wir Ihnen zusätzlich 50,- € auf Ihr Konto [TKT-00001].",
     "Rückerstattungen dauern 5 Werktage.", "de", "Bestellung 50 fehlt."),
    ("Als Entschädigung erstatten wir Ihnen zusätzlich 50.- EUR auf Ihr Konto [TKT-00001].",
     "Rückerstattungen dauern 5 Werktage.", "de", "Bestellung 50 fehlt."),
    ("As a goodwill gesture you get 50 pounds credit on your account [TKT-00001].", "Order 50 was refunded.", "en", "Refund."),
    ("Refunds are processed within 1 to 5 business days after we confirm [TKT-00001].", A5, "en", "Refund please."),
    ("Refunds are processed within 2-5 business days after we confirm [TKT-00001].", A5, "en", "Refund please."),
    ("Refunds take 5 business days and our hotline is open at 2 pm daily [TKT-00001].", A5, "en", "Refund please."),
    ("Als Entschädigung erhalten Sie eine zweiwöchige Testphase für den Tarif [TKT-00001].",
     "Rückerstattungen dauern 5 Werktage.", "de", "Erstattung bitte."),
    ("Rückerstattungen werden innerhalb von 14-tägiger Frist nach Eingang bearbeitet [TKT-00001].",
     "Rückerstattungen dauern 5 Werktage.", "de", "Erstattung bitte."),
])
def test_second_review_invented_quantities_fail(reply, answer, lang, ticket):
    r = verify({"reply": reply, "cited_ids": ["TKT-00001"], "unsupported_asks": []}, one(answer), ticket, lang)
    assert "numbers" in failed(r), r["checks"]


def test_range_inside_the_source_passes():
    r = check("Refunds are processed within 1 to 5 business days after we confirm [TKT-00001].",
              "Refunds take 1 to 5 business days.")
    assert "numbers" not in failed(r), r["failures"]


def test_article_duration_grounded_by_source_passes():
    r = check("Refunds are processed within a day after we confirm the return [TKT-00001].",
              "Refunds are processed within 1 day.")
    assert "numbers" not in failed(r), r["failures"]


def test_half_grounded_by_fifty_percent_passes():
    r = check("You will receive half of the amount back after the review [TKT-00001].",
              "We refund 50% of the amount after review.")
    assert "numbers" not in failed(r), r["failures"]


# 4: links and contact addresses in other shapes
@pytest.mark.parametrize("reply", [
    "Please visit help.example-shop.com/refunds to track your refund status [TKT-00001].",
    "Please write to refunds at example-shop dot com to track your refund [TKT-00001].",
    "Please write to billing(at)contoso.com for help with the refund today [TKT-00001].",
])
def test_second_review_links_and_emails_fail(reply):
    assert "placeholders" in failed(check(reply, A5))


def test_quoted_bare_domain_passes():
    r = check("You can track it on status.contoso.com at any time [TKT-00001].",
              "Track refunds on status.contoso.com.")
    assert "placeholders" not in failed(r), r["failures"]


# 5: dates with month names
@pytest.mark.parametrize("reply,answer,lang", [
    ("The new plan will start on March 3, 2027 as described in the notes [TKT-00001].",
     "The new plan will start on January 3, 2027.", "en"),
    ("Der neue Tarif startet am 3. März 2027 wie in den Versionshinweisen beschrieben [TKT-00001].",
     "Der neue Tarif startet am 3. Januar 2027.", "de"),
])
def test_wrong_month_name_date_fails(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


@pytest.mark.parametrize("reply,answer,lang", [
    ("The new plan will start on January 3, 2027 as described in the notes [TKT-00001].",
     "The new plan will start on 2027-01-03.", "en"),
    ("Der neue Tarif startet am 3. Januar 2027 wie in den Versionshinweisen beschrieben [TKT-00001].",
     "Der neue Tarif startet am 03.01.2027.", "de"),
    ("The new plan will start on January 3 as described in the notes [TKT-00001].",
     "The new plan will start on 2027-01-03.", "en"),
])
def test_month_name_date_matches_numeric_source(reply, answer, lang):
    r = check(reply, answer, lang)
    assert "numbers" not in failed(r), r["failures"]


# 6-8: realistic grounded replies that used to be blocked
@pytest.mark.parametrize("reply,lang,ticket", [
    ("Bitte warten Sie einige Minuten und setzen Sie dann das Passwort zurück [TKT-00001].", "de", "Passwort"),
    ("Es sind keine weiteren Benutzer betroffen. Bitte setzen Sie das Passwort zurück [TKT-00001].", "de", "Passwort"),
    ("Wenn ein Benutzer gesperrt ist, setzen Sie bitte das Passwort zurück [TKT-00001].", "de", "Passwort"),
    ("Bitte installieren Sie den Client auf jedem einzelnen Gerät erneut [TKT-00001].", "de", "Passwort"),
    ("This issue affects Windows 10 users after the latest update, please reset [TKT-00001].", "en",
     "Since the update my Windows 10 laptop can't log in."),
    ("The update fixes the login issue on iOS 17 devices, please reinstall the app [TKT-00001].", "en",
     "Login fails on my iOS 17 phone."),
    ("Das Problem betrifft Windows 11 Geräte nach dem Update, bitte neu anmelden [TKT-00001].", "de",
     "Seit dem Update geht Windows 11 nicht."),
    ("We have sent you a one-time password, please reset the password with it [TKT-00001].", "en", "Password"),
])
def test_second_review_false_blocks_pass(reply, lang, ticket):
    r = verify({"reply": reply, "cited_ids": ["TKT-00001"], "unsupported_asks": []},
               one("Bitte setzen Sie das Passwort zurück. Please reset the password."), ticket, lang)
    assert "numbers" not in failed(r), r["failures"]


def test_eines_tages_matches_one_day():
    r = check("Wir melden uns innerhalb eines Tages bei Ihnen zurück, versprochen [TKT-00001].",
              "Wir antworten innerhalb von 1 Tag.", "de")
    assert "numbers" not in failed(r), r["failures"]


def test_version_without_context_is_still_a_quantity():
    r = check("This issue affects Windows 12 users after the latest update [TKT-00001].", "Reset the password.")
    assert "numbers" in failed(r)


# 9: the source's language column is not trusted
def test_mislabeled_source_language_is_ignored():
    r = verify({"reply": "Ihr Tarif enthält 1.000 GB Speicher für alle Mitarbeiter im Team [TKT-00001].",
                "cited_ids": ["TKT-00001"], "unsupported_asks": []},
               [{"ticket_id": "TKT-00001", "answer": "Ihr Tarif enthält 1.000 GB Speicher.", "language": "en"}],
               "Speicher?", "de")
    assert "numbers" not in failed(r), r["failures"]


# 10: an ambiguous slash date must match exactly
@pytest.mark.parametrize("reply", ["Your contract ends on 02/01/2026 as stated in our records [TKT-00001].",
                                   "Your contract ends on 01/02/2026 as stated in our records [TKT-00001]."])
def test_ambiguous_slash_date_needs_exact_match(reply):
    assert "numbers" in failed(check(reply, "Your contract ends on 2026-02-01."))


def test_german_slash_date_is_day_first():
    r = check("Ihr Vertrag endet am 01/02/2026 wie in unseren Unterlagen vermerkt [TKT-00001].",
              "Your contract ends on 2026-02-01.", "de")
    assert "numbers" not in failed(r), r["failures"]


# 11: several short wrong-language sentences are caught together
def test_short_wrong_language_sentences_fail():
    r = check("Vielen Dank für Ihre Nachricht und Geduld. Your refund is approved now. We fixed your account "
              "today. Please check it again. Mit freundlichen Grüßen [TKT-00001].", A5, "de")
    assert "language" in failed(r)


# worst-case inputs stay fast
@pytest.mark.parametrize("reply", ["a" * 2999, "1 " * 1499, "1-" * 1499, "(617) 555-0199 " * 190])
def test_worst_case_replies_are_fast(reply):
    t = time.perf_counter()
    verify({"reply": reply, "cited_ids": ["TKT-00001"], "unsupported_asks": []}, one(A5), "ticket", "en")
    assert time.perf_counter() - t < 0.2


# ================================================================ regressions from the third cold review
EN_SRC = "Refunds of €20 are processed within 5 business days."
DE_SRC = "Rückerstattungen von 20 € werden innerhalb von 5 Werktagen bearbeitet."


# H1: magnitudes
@pytest.mark.parametrize("reply,answer,lang", [
    ("We will process your refund of €20k within 5 business days [TKT-00001].", EN_SRC, "en"),
    ("We will process your refund of €20 million within 5 business days [TKT-00001].", EN_SRC, "en"),
    ("We will process your refund of €20m within 5 business days [TKT-00001].", EN_SRC, "en"),
    ("Wir erstatten Ihnen €20 Mio. innerhalb von 5 Werktagen zurück [TKT-00001].", DE_SRC, "de"),
    ("Wir erstatten Ihnen zwanzig Millionen Euro innerhalb von 5 Werktagen [TKT-00001].", DE_SRC, "de"),
    ("Wir erstatten Ihnen 20 Tausend Euro innerhalb von 5 Werktagen zurück [TKT-00001].", DE_SRC, "de"),
    ("We will process your refund of 2 thousand euros within 5 business days [TKT-00001].", EN_SRC, "en"),
])
def test_magnitude_suffixes_are_read(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_magnitude_grounded_by_same_magnitude():
    r = check("The plan includes credits worth €20k per year for your team [TKT-00001].",
              "The plan includes credits worth €20,000 per year.")
    assert "numbers" not in failed(r), r["failures"]


# H2: billing period is part of the amount
@pytest.mark.parametrize("reply,lang", [
    ("The Pro plan costs €20 per month and includes 5 users for you [TKT-00001].", "en"),
    ("The Pro plan costs €20 monthly and includes 5 users for you [TKT-00001].", "en"),
    ("The Pro plan costs €20 per user and includes 5 users for you [TKT-00001].", "en"),
    ("The Pro plan costs €20 a month and includes 5 users for you [TKT-00001].", "en"),
])
def test_wrong_billing_period_fails(reply, lang):
    assert "numbers" in failed(check(reply, "The Pro plan costs €20 per year and includes 5 users.", lang))


def test_german_billing_period_fails_and_matches():
    src = "Der Pro-Tarif kostet 20 € pro Jahr und umfasst 5 Benutzer."
    assert "numbers" in failed(check("Der Pro-Tarif kostet 20 € pro Monat und umfasst 5 Benutzer [TKT-00001].", src, "de"))
    r = check("Der Pro-Tarif kostet 20 € jährlich und umfasst 5 Benutzer für Sie [TKT-00001].", src, "de")
    assert "numbers" not in failed(r), r["failures"]


# H3 / H4: countable things outside the old unit list, and spelled-out fractions
@pytest.mark.parametrize("reply,answer,lang", [
    ("Refunds take 5 business days and we waive your fee for two billing cycles [TKT-00001].", EN_SRC, "en"),
    ("Refunds take 5 business days and support is open three weekends per year [TKT-00001].", EN_SRC, "en"),
    ("Refunds are processed within a fortnight after we confirm the return [TKT-00001].", EN_SRC, "en"),
    ("Refunds are processed within one and a half hours after we confirm it [TKT-00001].",
     EN_SRC + " A reset takes half an hour.", "en"),
    ("Die Rückerstattung erfolgt innerhalb von zweieinhalb Tagen auf Ihr Konto [TKT-00001].", DE_SRC, "de"),
    ("Die Rückerstattung erfolgt innerhalb von anderthalb Stunden auf Ihr Konto [TKT-00001].", DE_SRC, "de"),
    ("Die Rückerstattung erfolgt innerhalb von eineinhalb Wochen auf Ihr Konto [TKT-00001].", DE_SRC, "de"),
    ("Die Rückerstattung erfolgt innerhalb einer Viertelstunde auf Ihr Konto zurück [TKT-00001].", DE_SRC, "de"),
    ("Die Rückerstattung erfolgt innerhalb eines Werktages auf Ihr Konto zurück [TKT-00001].", DE_SRC, "de"),
    ("Die Rückerstattung erfolgt innerhalb von drei Geschäftstagen auf Ihr Konto [TKT-00001].", DE_SRC, "de"),
    ("Refunds take 5 business days and the new storage is tenfold larger [TKT-00001].", EN_SRC, "en"),
])
def test_third_review_invented_quantities_fail(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_one_and_a_half_matches_source():
    r = check("Refunds are processed within one and a half hours after we confirm it [TKT-00001].",
              "Refunds are processed within 1.5 hours.")
    assert "numbers" not in failed(r), r["failures"]


# H5 / H6 / L1: links and spelled-out emails
@pytest.mark.parametrize("reply", [
    "Please reset it at help.refundcenter.ai/reset now, then wait [TKT-00001].",
    "Please reset it at secure-login.ru now, then wait for the email [TKT-00001].",
    "Please reset it at hilfe.bücher.de now, then wait for the email [TKT-00001].",
    "Please reset it at pаypal.com/login now, then wait for the email [TKT-00001].",
    "Please reset it at example.co now, then wait for the confirmation [TKT-00001].",
    "Please reset it at https://example.com/he now, then wait for it [TKT-00001].",
    "Please write to refunds {at} shop {dot} com for the refund status [TKT-00001].",
])
def test_third_review_links_fail(reply):
    r = check(reply, "Reset it at https://example.com/help within 5 business days.")
    assert "placeholders" in failed(r)


@pytest.mark.parametrize("reply", [
    "Please reset it at https://example.com/help now, then wait for it [TKT-00001].",
    "Please reset it on example.com now, then wait for the confirmation [TKT-00001].",
    "Please send us the setup.log and the config.yaml from the client [TKT-00001].",
])
def test_quoted_links_and_file_names_pass(reply):
    r = check(reply, "Reset it at https://example.com/help within 5 business days.")
    assert "placeholders" not in failed(r), r["failures"]


# H7: ticket numbers can't become promised amounts; IDs still pass
def test_bare_amount_after_money_word_needs_a_source():
    r = check("We will process your refund of 500 within 5 business days for you [TKT-00001].", EN_SRC,
              ticket="I was charged 500 twice, please refund.")
    assert "numbers" in failed(r)


@pytest.mark.parametrize("reply", [
    "We have processed the refund for order 12345 within 5 business days [TKT-00001].",
    "We have credited invoice no. 4711 within 5 business days as promised [TKT-00001].",
])
def test_ids_after_money_words_stay_plain(reply):
    r = check(reply, EN_SRC, ticket="Order 12345 and invoice no. 4711 were charged twice.")
    assert "numbers" not in failed(r), r["failures"]


def test_three_word_gap_unit_from_ticket_number_fails():
    r = check("We add 30 additional premium support days to your plan today [TKT-00001].", EN_SRC,
              ticket="Error 30 appears when I log in.")
    assert "numbers" in failed(r)


# H8: numbers must come from the tickets the draft cites
def test_number_from_uncited_retrieved_row_fails():
    rows = [{"ticket_id": "TKT-00001", "answer": EN_SRC},
            {"ticket_id": "TKT-00002", "answer": "Enterprise plans cost €500 per year."}]
    r = verify({"reply": "We will process your refund of €500 within 5 business days [TKT-00001].",
                "cited_ids": ["TKT-00001"], "unsupported_asks": []}, rows, "Refund please.", "en")
    assert "numbers" in failed(r)


# M1: long tickets and sources stay fast
@pytest.mark.parametrize("ticket", ["1.1.1." * 2000, "12,34," * 1000, "1,1," * 3000, "1.1.1." * 8000])
def test_long_number_runs_in_ticket_are_fast(ticket):
    t = time.perf_counter()
    check("We will process your refund of €20 within 5 business days [TKT-00001].", EN_SRC, ticket=ticket)
    assert time.perf_counter() - t < 1.0


# M2: a word that merely contains a unit is not a unit
def test_today_is_not_a_day_count():
    r = check("Your replacement order 48213 ships today, within 5 business days at most [TKT-00001].", EN_SRC,
              ticket="Order 48213 arrived broken.")
    assert "numbers" not in failed(r), r["failures"]


# L1: superscript and full-width digits are read as digits
@pytest.mark.parametrize("reply", ["Your new plan includes ²⁰⁰ GB of storage, refunds in 5 business days [TKT-00001].",
                                   "Your new plan includes ２００ GB of storage, refunds in 5 business days [TKT-00001]."])
def test_compatibility_digits_are_read(reply):
    assert "numbers" in failed(check(reply, EN_SRC))


def test_double_charge_is_not_a_multiplier():
    r = check("We fixed the double charge, refunds take 5 business days [TKT-00001].", EN_SRC)
    assert "numbers" not in failed(r), r["failures"]
