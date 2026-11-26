# SentinelDesk

**A support-ticket copilot that never misses an outage and never makes up an answer.** English and German.

For every incoming ticket, SentinelDesk:

1. **Triages** it: type, queue and priority, each with a calibrated confidence.
2. **Retrieves** the most similar resolved past tickets (hybrid BM25 + multilingual dense search).
3. **Answers one question with a YES/NO checklist:** *may the AI write this reply?*
4. **Drafts** a reply from the retrieved answers only, citing them (`[TKT-01234]`).
5. **Verifies** the draft with deterministic rules before anyone sees it. Any failure escalates.
6. **Escalates** everything else to the right human, with the reasons, a templated acknowledgement for the customer, and a handoff record.

```
ticket ─► mask PII ─► language ─► triage ──────────────┐
                          └────► hybrid retrieval ─────┤
                                                       ▼
                                  YES/NO gate (Q1–Q7): critical outage? confident? close precedent?
                                  unsupported ask? injection? ─── any NO ──► escalate (no LLM call)
                                                       │ all YES
                                                       ▼
                                  LLM draft (JSON: reply, cited_ids, unsupported_asks)
                                                       ▼
                                  ⑤ VERIFY (V1–V6, rules, ~1 ms) ─── any NO ──► escalate, draft blocked
                                                       │ all YES
                                                       ▼
                                               🟢 Auto-Draft Ready
```

## Origin and credits

SentinelDesk started at the **GDG Cloud Boston × Northeastern hackathon** (November 2026) as a six-person team project: [wuhong4088/HackathonGDG](https://github.com/wuhong4088/HackathonGDG). In the team build, I owned the **verify step (⑤)** ([PR #5](https://github.com/wuhong4088/HackathonGDG/pull/5)). Teammates built the team's triage ensemble, retrieval, Gemini drafting, pipeline and console.

**This repository is my own end-to-end implementation**: triage, retrieval, the YES/NO gate, the escalation ladder, burst detection, the console and the evaluation scripts. `sentineldesk/verify.py` and its tests are the module I wrote for the team, adapted here to this codebase's own PII and language modules. No teammate code is included.

## Highlights

### ⑤ Verify: rules that check the LLM
LLM output is never trusted. Six checks run on every draft, and **all of them** must pass:

| Check | Fails when |
|---|---|
| `format` | The reply is empty or too long, or the JSON is malformed. Never crashes on bad input |
| `citations` | A cited ticket wasn't retrieved, or nothing is cited |
| `numbers` | An amount, %, duration, date or time isn't in a **cited** answer with the same unit and billing period |
| `placeholders` | An invented email, phone, account number or link (any TLD, incl. Unicode lookalikes) |
| `language` | The reply, or any run of its sentences, isn't in the customer's language |
| `unsupported_asks` | The LLM listed a promise it couldn't ground (routed to the policy owner) |

The number check understands English and German formats and number words: `1.000,50 €` = `1000.50 EUR`, "vierundzwanzig Stunden" = 24 hours, "zweieinhalb Tage", "an hour", "€20k", "20 Mio.", "per month" vs "per year", and ranges. A bare "refund of 500" counts as an amount, while "order 12345" stays a plain number.

**How it was hardened:** three rounds of independent adversarial review found 30+ ways an invented value or link could slip through, plus a worst case where a pasted CSV row made the check take **33 s (now 0.09 s)**. Each case became a regression test.
- **161 tests**
- **~1 ms** per draft
- **2 false blocks on 1,000 real support answers** checked against themselves (0.2%, both from language detection on mixed-language answers)

### A YES/NO gate instead of a black box
Every question is phrased so that YES means safe, and each NO carries its evidence:

```
ai_may_respond: NO   first NO: Is it free of signs of a critical outage?
 Q4 NO   Is it free of signs of a critical outage?   outage words: emr, is down, offline
 Q6 YES  Is there a close past ticket?               best similarity 0.71 (need 0.55)
→ ESCALATE · P1 → On-call engineer
```

### Escalation ladder (with or without severity)
`SENTINEL_ESCALATION_MODE=severity` (the default) routes by severity. `simple` sends every escalation to on-call, with all reasons listed.

| Rung | Owner | Triggered by |
|---|---|---|
| L4 / P1 | Major Incident | 3+ similar Incident/Problem tickets in 15 min (burst detection) |
| L3 / P1 | On-call engineer | High-priority Incident, or outage language |
| L2 / P2 | Policy owner (legal → billing → sales) | Refund, price, SLA or legal requests |
| L1 / P3 | Queue team or human triage desk | No precedent, low confidence, failed verify, bad input |

The customer gets a fixed English or German acknowledgement with an `ESC-xxxx` reference, never LLM text. Agents get a handoff record that includes the full checklist.

## Results

20% holdout (5,718 tickets). Thresholds and weights were tuned on a validation slice of the 80% knowledge base, never on the holdout.

| Triage | Macro-F1 | Classifier only | Majority baseline |
|---|---|---|---|
| type | **0.891** | 0.866 | 0.142 |
| queue (10 classes) | **0.652** | 0.577 | 0.045 |
| priority | **0.707** | 0.591 | 0.192 |

- **High-priority Incident recall: 90.9%** paged to on-call directly (precision 52%), **94.2%** escalated by some route. The P(high) threshold is set low on purpose: a false page costs less than a missed outage.
- **Latency before the LLM:** 12.7 ms p50, 20.1 ms p95.

Classifier: TF-IDF (word + character n-grams) with a calibrated LinearSVC, blended 50/50 with the labels of the top-5 retrieved precedents.

**Retrieval models, measured on this data** (1,000 validation queries; "top-1 same queue" means the closest past ticket is in the query's queue):

| Model | Top-1 same queue | MRR@10 | Query time |
|---|---|---|---|
| **bge-m3** | **0.845** | **0.882** | 43 ms |
| paraphrase-multilingual-mpnet-base-v2 | 0.814 | 0.861 | 10 ms |
| paraphrase-multilingual-MiniLM-L12-v2 (used here) | 0.808 | 0.856 | 2.5 ms |
| TF-IDF / BM25 | 0.748 / 0.746 | 0.81 / 0.81 | < 1 ms |
| multilingual-e5-small / -base / -large-instruct | 0.734 / 0.721 / 0.736 | ~0.79 | 2–14 ms |

The e5 models rank well on public leaderboards but score below plain TF-IDF on this paraphrase-heavy data. Plain RRF hybrid search (0.777) also scored *below* dense search alone (0.816). Both findings are reproducible with `scripts/bench_embeddings.py` and `scripts/compare_retrieval.py`.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train            # download the data, lock the 80/20 split, train triage, build the indexes
python -m scripts.evaluate         # holdout metrics -> reports/eval_report.json
python -m scripts.smoke_test       # demo scenarios in both escalation modes
python -m pytest tests -q          # the 161 verify tests
streamlit run app.py               # the operator console
```

Set `ANTHROPIC_API_KEY` for LLM drafts. Without it, drafts fall back to quoting the closest past answer verbatim, and the console still works.

## Data

[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), file `aa_dataset-tickets-multi-lang-5-2-50-version.csv`: 28,587 synthetic English and German IT-support tickets, licensed **CC-BY-NC-4.0** (non-commercial). The data is downloaded by `scripts/train.py` and is **not** included in this repo.

Things to know about this data:
- The `language` column is wrong on about 11% of rows, so language is always detected from the text.
- 13% of subjects are empty.
- Many tickets are reworded near-duplicates, which inflates retrieval scores.

## Limitations
- Replies under 8 words aren't language-checked.
- Invented person names can't be caught by rules, so the draft prompt has to forbid them.
- The outage keyword list is hand-made; a trained critical detector would be better.
- "Euro 2025" is read as an amount (it errs on the safe side).

## License
Code: Apache-2.0 (see `LICENSE`). Data: CC-BY-NC-4.0, by its authors.
