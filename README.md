# SentinelDesk

Hackathon project for the GDG Cloud Boston × Northeastern hackathon.

SentinelDesk takes an incoming support ticket (English or German) and:

1. works out what the ticket is and how urgent it is,
2. finds similar tickets we resolved in the past,
3. decides whether the AI may answer or a human must take it.

## Status board

_Last updated 14:15. Update this table in the same commit as your work: change the status, add your name. Run `python check_rules.py` before every push._

| Status | Piece | File | Owner | Notes |
|---|---|---|---|---|
| ✅ Done | Plan, architecture, RAG plan, leakage rules | README.md | | |
| ✅ Done | Fair-play and scope rules (loaded by everyone's Claude) | CLAUDE.md | | |
| ✅ Done | Locked 80/20 split + leakage check | split.py, check_rules.py | | Uses the main CSV only (22,869 KB / 5,718 holdout). Check prints `OK` |
| ✅ Done | PII masking (bonus) | pii.py | | Self-check: `python pii.py`. Must become step 1 of the pipeline |
| ✅ Done | Language detection | sentineldesk/language.py | Hung | ~0.02 ms/ticket, zero deps. Removed German IT loanwords (`account`, `system`, `error`, `access`, `update`, `support`, `customer`, `team`) from `EN_PROFILE` and cut unused `fr`/`es` profiles |
| ✅ Done | Data audit: source choice, missing values, broken rows, reworded copies, label noise | docs/DATA_PREPROCESSING.md | | 11 suggested changes, high priority first |
| ✅ Done | RAG plan + locked `search()` interface | README.md ([RAG plan](#rag-plan)) | Chirag | **Approved by the team.** Model: multilingual-e5-base, σ = 0.86. Code against `search()` now |
| ✅ Done | Hybrid retrieval (BM25 + dense + RRF) | retrieve.py | Chirag | `from retrieve import search`. Self-check: `python retrieve.py check`. Needs the shared index from `python scripts/embed.py`. σ = 0.86. See [Retrieval results](#retrieval-results) |
| ✅ Done | Triage: critical detector → TF-IDF model → neighbour votes → ensemble + calibration | sentineldesk/triage/, scripts/*_triage.py | SPGOHIL09 | `Triage.load("artifacts").predict(subject, body)`; see sentineldesk/triage/README.md. Trains in ~6 min on a laptop (or Colab: notebooks/triage_colab.ipynb) and builds the shared `data/index/` (e5-base). Applied the data audit: family-based half A/B, `\n`/`<br>` cleaning, known vs novel holdout slices; ensemble = weighted average (beat the LogReg combiner). Neighbour search uses NumPy, **not FAISS**: FAISS loaded after PyTorch segfaults/hangs on macOS |
| ✅ Done | Pipeline glue: `Pipeline().handle(subject, body)` → everything the console shows | pipeline.py | Chirag | mask → language → triage → search → gate → draft → verify. ~25–30 ms without the LLM. Self-check: `python pipeline.py`. Drafter is pluggable: `Pipeline(drafter=fn)`, `fn(text, hits, lang) -> {reply, cited_ids, unsupported_asks}`. Uses `sentineldesk.verify` once PR #5 merges |
| 🔨 First version | Escalation gate (rules + reason strings) | pipeline.py `gate()` | Chirag (until someone takes it) | Escalates on: critical detector fires, high-priority Incident, low type/priority confidence, best similarity < σ, unsupported asks, failed verify. Low **queue** confidence only flags "routing uncertain" + top-2 queues |
| ✅ Done | Draft (step ④: LLM writes the reply from retrieved answers, JSON output) | sentineldesk/draft.py | Chirag | `draft(ticket_text, hits, lang)` → `{reply, cited_ids, unsupported_asks, model, latency_ms}`; raises `DraftError` (→ escalate "drafting unavailable"). Feeds straight into `verify()`. Live test: 5/5 normal en/de tickets pass verify in ~1.5 s, 2/2 refund/prompt-injection attacks escalate. Needs `GEMINI_API_KEY` |
| 🔨 In progress | Verify (step ⑤: rule checks on every LLM draft; fails → escalate) | | Aashay | PR #5 open, not merged yet |
| 🔨 In progress | Live console UI | | Sudeep | |
| 🔨 Ready, not run | Eval script: Macro-F1 + high-priority Incident recall, "ML only" vs "ML + rules", latency | scripts/eval_triage.py | SPGOHIL09 | Triage part written and tested on a sample. Run on the holdout **once**, at the end |
| ✅ Done | requirements.txt | requirements.txt | SPGOHIL09 | scikit-learn pinned to 1.7.2 (models are pickled on Colab, loaded on the laptop) |
| ⬜ To do | Demo script + rehearsal | | | Last hour |
| ⬜ Bonus | Cross-lingual matching (comes from the dense embeddings), burst detection | | | Only after the MVP works |

**Blocking decisions** (see [Open decisions](#open-decisions)): the LLM, the UI framework, team roles, what happens after "Escalate", and the similarity threshold.

## The problem

Helpdesks lose time in two ways:

- **Missed outages.** A critical outage (for example, a VPN failure that cuts hospital staff off from patient records) lands in the same inbox as "what does your product cost?" and waits hours.
- **Made-up answers.** Generic AI chatbots write confident replies that promise refunds, SLAs or features nobody agreed to.

Our goal is to **never miss an outage and never make up an answer.**

## Agreed workflow

Steps ①–③ are decided. Everything else is still open (see [Open decisions](#open-decisions)).

```
   👤 Customer sends a ticket (subject may be empty + body, en or de)
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  ① UNDERSTAND IT                               │
│  • What language? (en / de)                    │
│  • What kind?   Incident / Request /           │
│                 Problem / Change               │
│  • Which team?  Tech Support, Billing,         │
│                 IT, Sales, ...                 │
│  • How urgent?  high / medium / low            │
│  • How sure are we? (confidence %)             │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│  ② LOOK UP THE PAST                            │
│  "Have we solved something like this before?"  │
│  → find 3-5 most similar resolved tickets      │
│    and how agents answered them                │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│  ③ SAFETY CHECK: is it OK for AI to answer?   │
│                                                │
│   Escalate to a human if ANY is true:          │
│   ✗ urgent outage (critical score is high)     │
│   ✗ system isn't confident in its triage       │
│   ✗ no similar past ticket exists              │
│   ✗ customer asks for something we've never    │
│     done (refund promise, new feature, ...)    │
└───────────┬───────────────────────┬───────────┘
        risky│                       │safe
            ▼                       ▼
      🔴 Escalate              🟢 AI may draft
      (details TBD)            (details TBD)
```

## Models (decided)

Only decisions the team has agreed on. The drafting LLM is still open (see [Open decisions](#open-decisions)).

| Part | Model | Notes |
|---|---|---|
| Embeddings (retrieval, classifier features, neighbour votes, burst detection) | `intfloat/multilingual-e5-base` | One model for everything. Switching means re-embedding the index, retraining the classifier and re-picking σ |
| Keyword search | BM25 (no model) | Fitted on the 80% split only |
| Triage | Ensemble: LogReg on e5-base embeddings + TF-IDF LogReg + neighbour votes, calibrated, plus a critical detector | See [Triage in detail](#triage-in-detail-options-0-1-2-5-6). No LLM in triage |
| Language detection | `sentineldesk/language.py` (team's own en/de detector) | Never read the dataset's `language` column at inference |

## Architecture (hybrid)

We use classical ML where results must be measurable, rules where we can't afford to miss something, and the LLM only to write the reply. The LLM's output is then checked before anyone sees it.

```
 Ticket (subject?, body)
        │
        ▼
 ① TRIAGE  (details in "Triage in detail" below)
   ├─ Ticket text → embedded once (shared with retrieval) + TF-IDF once
   ├─ Three models, each gives probabilities for type / queue / priority:
   │    (0) multilingual embeddings → LogReg ×3
   │    (1) TF-IDF (word + character n-grams) → LogReg ×3
   │    (2) neighbour votes: labels of the top-20 most similar past tickets
   ├─ (6) Ensemble: weighted average of (0) + (1) + (2) log-probs (3 weights/task, fit on half A) → calibrated confidence
   └─ (5) Critical detector: embeddings + outage-keyword flag (en + de)
          → score = how likely this is a high-priority Incident
          → feeds the gate; it does NOT overwrite the predicted labels
        │
        ▼
 ② RETRIEVAL (hybrid search)
   ├─ BM25 (exact terms: product names, error codes)
   ├─ Dense multilingual embeddings (meaning; German ↔ English)
   └─ Reciprocal Rank Fusion → top 3–5
        │
        ▼
 ③ GATE (deterministic rules, each with a reason string)
   critical score │ low confidence │ low similarity │ unsupported ask
        │                                         │
     escalate                                   safe
        │                                         ▼
        │                     ④ DRAFT: LLM, given only the retrieved answers
        │                        returns JSON {reply, cited_ids, unsupported_asks}
        │                                         ▼
        │                     ⑤ VERIFY (rules)
        │                        • cited_ids ⊆ retrieved ticket_ids
        │                        • every number / € / $ / % / SLA term ("within X days")
        │                          in the reply must also appear in a retrieved answer
        │                        • keep placeholders like <name>, <acc_num> as they are
        │                        • check fails → escalate instead of showing the draft
        ▼                                         ▼
  🔴 Escalate to On-Call (reason)        🟢 Auto-Draft Ready
```

### Design rules

- **The LLM does not do triage.** LLM confidence scores aren't calibrated, and LLM calls would make the holdout evaluation (about 5.7k tickets) slow, rate-limited and non-deterministic.
- **The UI shows the dense cosine similarity** and the "no precedent" threshold uses it. The RRF score is only used to order results, because it is not a similarity.
- **Fit everything on the 80% split only:** BM25 IDF statistics, the embedding index, the classifiers and calibration.
- **Build the outage keyword list from the 80% split** (its `Outage`/`Disruption` tags) and from common sense. Never tune it by looking at holdout errors. The list is an input to the critical detector, not a rule that changes labels.
- **Keyword matches never overwrite predicted labels.** The holdout labels are noisy, so forcing labels would lower Macro-F1. Outages are handled by escalation instead.
- **Report two sets of metrics:**
  - Macro-F1 for "current model only (0)" vs "full triage (0 + 1 + 2 + 6)".
  - High-priority Incident recall and precision for "predicted labels only" vs "with critical detector (5)".

  The detector raises recall and may cost some precision, and we show that trade-off on purpose.
- **Mask PII before any external LLM call** (bonus): emails, phone numbers, IPs and account numbers.

## Triage in detail (options 0, 1, 2, 5, 6)

We keep the original idea (embeddings + logistic regression) as the base and add four pieces around it.

| # | Piece | What it adds |
|---|---|---|
| 0 | Embeddings + LogReg (original idea) | Understands meaning, works across German and English |
| 1 | TF-IDF + LogReg | Catches exact words: product names, error codes |
| 2 | Neighbour votes | "8 of 10 similar tickets went to Billing": no training needed, and gives an explanation for the UI |
| 6 | Ensemble + calibration | Combines 0, 1 and 2 with 3 weights per task picked on half A (a weighted average of log-probabilities; it beat the planned small LogReg by +5 points mean Macro-F1 on new tickets), then temperature scaling on half B |
| 5 | Critical detector | Targets high-priority Incident recall with a threshold we can set; replaces the old keyword overrides |

**At prediction time:** the ticket is embedded once and turned into TF-IDF once. Models 0, 1 and 2 each give probabilities, and the ensemble (6) turns them into one calibrated answer per task. The critical detector (5) gives a separate score. In the gate:

- the critical score decides "escalate to on-call",
- the ensemble confidence decides "low confidence",
- the retrieval cosine similarity decides "no precedent".

### Training, in order

1. **Split the 80% data:** 90% training part, 10% calibration slice. Cut the slice into two halves, **A** and **B** (about 1,200 tickets each). Use `random_state=42` everywhere. **Draw them by paraphrase family** (tickets with embedding cosine ≥ 0.98 are reworded copies): no half-A / half-B ticket has a copy in the training part, so calibration and thresholds are tuned on tickets that look new (`docs/DATA_PREPROCESSING.md` items 3–4).
2. **Compute embeddings once** for the whole 80% split and save them. Retrieval, model 0, neighbour votes and the critical detector all reuse them.
3. **Train models 0 and 1 on the training part.** Neighbour votes (2) need no training; they search the training part.
4. **Train the ensemble (6) on half A,** using the outputs of models 0, 1 and 2 on half-A tickets.
5. **Calibrate the ensemble on half B** and draw the reliability chart.
6. **Train the critical detector (5) on the training part** (embeddings + outage-keyword flag). Tune its threshold on half B to catch at least 90% of high-priority Incidents.
7. **Run the holdout once,** at the very end, for the final numbers.

### Two mistakes to avoid

- **Don't train the ensemble on the training part.** Models 0 and 1 have already seen those tickets and would look over-confident.
- **Don't let a ticket vote for itself.** For half-A tickets, neighbour votes search only the training part. Add a check that a ticket never appears in its own neighbour list. Its reworded copies must not vote either: the family-based split guarantees none of them is in the training part.

### Build order and fallback

Build in this order, so stopping early still leaves something useful:

1. Critical detector (5): ~20–30 min of building
2. TF-IDF model (1): ~20–30 min
3. Neighbour votes (2): ~30 min, by the retrieval owner
4. Ensemble + calibration (6): ~45–60 min

If we're behind at the 3:30 checkpoint, drop the ensemble and keep model 0 + the critical detector.

**Shared format:** every model returns probabilities per task in the same, fixed label order. Agree on this at the first checkpoint.

### Training time on Google Colab (free T4 GPU)

| Step | Runs on | Time |
|---|---|---|
| Setup (packages, data, model download) | — | ~3–6 min |
| Embeddings for ~28.6k tickets | GPU | ~4–8 min (bge-m3) / ~1–2 min (e5-small) |
| Model 0 | CPU | ~2–5 min |
| Model 1 (TF-IDF) | CPU | ~5–12 min |
| Models 2, 5, 6 | CPU | under 1 min each |
| **Total** | | **~15–30 min (bge-m3) / ~10–20 min (e5-small)** |

These are estimates. Save embeddings and models to Google Drive so a disconnect doesn't mean re-embedding. Train in Colab, then run the demo on a laptop from the saved files.

## RAG plan

**Knowledge base:** one entry per past ticket from the 80% split: `ticket_id`, subject + body, answer, language, type, queue, priority, tags.
- **Drop junk answers:** empty ones, ones under about 40 characters, and ones that are only placeholders (for example "Den Fall prüfen, `<tel_num>` anrufen…"). They give the LLM nothing to ground on.

**What we search on:** subject + body. Answers are not searchable, because we match on the customer's problem, not on how an agent phrased the fix.

**PII mask first:** every query goes through `pii.mask()` before it is embedded or searched. KB text already uses `<email>`, `<tel_num>` and `<acc_num>` placeholders, so a masked query looks like KB text, while a raw email address would only add noise.

**Using categories.** A new ticket arrives with no category, only our classifier's prediction (type is right about 87% of the time, queue only about 50%), so:
- ✅ **Soft boost (default).** After searching, move up past tickets whose type/queue matches the prediction, by an amount scaled by the classifier's confidence. A low-confidence guess barely changes anything.
- ✅ **Same-language boost.** A small boost for past tickets in the query's language. Other-language tickets stay eligible, so a German ticket can still use English precedents (cross-lingual bonus).
- ✅ **Categories go to the LLM.** Each precedent is passed with its queue/type so the LLM can judge how relevant it is.
- 🧪 **Experiment: tags in keyword search only.** Append each past ticket's frequent tags (the 96 used 50+ times) to its **BM25** text. Embeddings stay on the raw text, so the shared embeddings file doesn't change. It's an on/off switch, kept only if the self-check shows it helps. See [Tag numbers](#tag-numbers). **Result: OFF.** No measurable gain (type@5 0.764 vs 0.763, queue@5 equal, cross-lingual within noise on 30 probes).
- ❌ **No hard filter by predicted queue.** It would hide the right match about half the time.

**Remove near-duplicate results:** if two of the top 5 have almost identical answers (text similarity > 0.9, `difflib`), keep the higher-ranked one and take the next result instead. Returned answers go through `pii.normalize_placeholders()`, so `[Name]`/`{tel_num}` become `<name>`/`<tel_num>` before the LLM sees them.

**"No precedent" threshold σ:** if the top result's dense cosine similarity is below σ, escalate. Pick σ on a slice of the 80% split, as the gap between in-domain tickets and off-topic probes (recipes, sports, prompt injection, in en and de). Use one σ for both languages unless the data shows they differ. **σ depends on the embedding model** (e5 scores cluster high, bge-m3 is more spread out), so re-pick it if the model changes.

**Pipeline order:** mask PII, then classify, then search (so the boost can use the predictions), then the safety check.

**How we evaluate retrieval (on a slice of the 80% split, never the holdout):**
- Self-retrieval: a past ticket searched by its own text must come back first.
- How often the top 5 share the query ticket's type and queue. This is only a sanity check, because the boost inflates it by design.
- Cross-lingual: translate some KB tickets into the other language, search with the translation, and check that the original comes back.
- Hand-check about 20 tickets: is the top answer actually useful?
- Search latency p50/p95.

**Handoff to the LLM:** `[{ticket_id, similarity, language, type, queue, answer}]`, top 5.

### Tag numbers

Measured on the 80% split (22,869 tickets):

| Finding | Number | What we do with it |
|---|---|---|
| Every ticket has tags | 1–8 per ticket, median 5 | Tags are usable for every past ticket |
| Tags are English even on German tickets | 1 of ~107k tag uses has an umlaut | Tags give the same English label to tickets in both languages. Mostly helps **English queries find German tickets** through keyword search |
| Tags add new words | 67% of tags don't appear in the ticket text | Worth testing in keyword search |
| Long tail | 1,138 distinct tags; 539 used once; 96 used 50+ times; top 30 = 88% of uses | Use only the 96 frequent tags |
| Predict type | NMI 0.18 | Some help |
| Predict queue | NMI 0.12 overall, but Billing → Billing queue 99%, Payment 100%, Account 77% | A few tags are near-certain |
| Predict priority | NMI 0.01–0.03 | No help; escalation stays the safety net |
| Outage-type tags | Crash: 60% high priority, 69% Incident. Outage: 53% / 71%. Disruption: 48% / 69% (average ticket: 39% / 40%) | **Critical detector:** build the outage keyword list from Outage, Disruption and Crash tickets |
| Request-type tags | Feature: 10% Incident. Sales: 9% Incident | |

New tickets have **no tags**, so tags are only ever used on the past-ticket side. **UI idea:** show the most common tags among the retrieved past tickets as likely topics (e.g. "Outage · Network"). **Burst detection (bonus):** several recent tickets whose neighbours are tagged Outage or Crash is a strong "major incident" signal.

### 🔒 Locked interface: `search()`

This is the one function everyone calls to use retrieval. Code against it now, with fake results if retrieval isn't ready yet.

```python
search(text, k=5, preds=None, lang=None, exclude_ids=None, pool_ids=None)
  -> [{"ticket_id", "similarity", "language", "type", "queue", "priority", "subject", "answer"}, ...]  # best first
```

| Input | Meaning | Who passes it |
|---|---|---|
| `text` | New ticket: `subject + "\n" + body`, already PII-masked | everyone |
| `k` | Number of past tickets to return | 5 for RAG, 10 for neighbour votes |
| `preds` | Classifier probabilities, e.g. `{"type": {"Incident": 0.91, ...}, "queue": {...}}`. Drives the soft boost | pipeline (after triage); `None` for neighbour votes |
| `lang` | `en`/`de` from `sentineldesk/language.py`. Drives the same-language boost | pipeline |
| `exclude_ids` | Ticket IDs never to return (a ticket must not find itself) | ML person, neighbour votes only |
| `pool_ids` | Search only these tickets (for example the 90% training part during calibration) | ML person, neighbour votes only |

- **`similarity`** is the dense cosine similarity, never the RRF score. The gate escalates "no precedent" when `retrieve.top_similarity(hits) < retrieve.SIGMA` (σ = 0.86). That's the **highest** similarity among the results, not rank 1's: the boosts can reorder results.
- **Optional `q`:** pass triage's embedding of the same text (`search(text, ..., q=vec)`) so the query is embedded once.
- **Neighbour votes** use **dense-only** search (no BM25, no boosts), weighted by `similarity`. Boosting by triage predictions would be circular.
- **Live demo:** nobody passes `exclude_ids` or `pool_ids`.

### Shared embeddings file (built by `python scripts/embed.py`)

Retrieval and the classifier share one file. `scripts/build_split.py` then `scripts/embed.py` (the team model from `configs/triage.yaml`) write it with these settings; `retrieve.py` refuses to load an index whose `meta.json` names a different model. **Rebuild it whenever `split.py`'s `text` column changes**, or live queries won't match the stored text:

- **Model:** `intfloat/multilingual-e5-base`
- **Text:** the `text` column from `split.py` (`subject + "\n" + body`, missing subject → empty)
- **Prefix:** `"query: "` before every text, past tickets included
- **Encode with** `normalize_embeddings=True`, saved as float32
- **Tickets:** all 22,869 from the 80% split (`load_split()[0]`), in that row order
- **Outputs, in `data/index/`** (gitignored; share them through Drive):
  - `emb.npy`: shape `(22869, 768)`
  - `kb.pkl`: same row order, with `ticket_id, text, subject, answer, language, type, queue, priority, tag_1..tag_8`, plus `searchable` (from `sentineldesk/triage/data.py:is_searchable`: False for empty, placeholder-only or under-40-character answers, and for the data audit's broken rows: agent-voice opener, body/answer word Jaccard ≥ 0.60, under 6 words. 22,486 of 22,869 are searchable)
  - `meta.json`: model + prefix
- **Run `python check_rules.py` after building.** It verifies the index holds only 80%-split tickets.

### Retrieval results

`python retrieve.py check`, on the 80% split only (300 queries from half A, each excluded from its own results), plus 13 hand-typed tickets and 12 off-topic probes written for the check:

| Check | Result |
|---|---|
| Self-retrieval, exclusion, junk filter | ✅ 20/20, never returns excluded or non-searchable tickets |
| Type / queue match in top 5 (sanity check only) | 0.77 / 0.45 |
| Cross-lingual: translated ticket finds its original, other-language tickets only | EN→DE 68%, DE→EN 97% (top 5) |
| Same, in full search | 29% / 13%. Same-language near-duplicates usually rank first; that's fine, they are valid precedents |
| Top-1 similarity | dataset tickets p50 0.976 · **hand-typed min 0.870, p50 0.886** · **off-topic max 0.853** |
| σ = 0.86 | all 12 off-topic probes below, 0 of 13 hand-typed below. **The gap is only ~0.01**, so the gate should also use the LLM's out-of-scope flag |
| Latency | search 6 ms p50 / 9 ms p95; query embedding ~8 ms warm on a Mac (first call ~1 s: warm up at app start) |

**Fixes found while testing:**
- **Keyword search needs stopwords and half weight.** In this synthetic corpus, filler words like "seit" and "Kontakt" are rare, so BM25 ranked a billing ticket first for a German VPN outage. With en/de stopwords and BM25 at 0.5 weight in RRF, that query returns "VPN Connection Issue" (an English precedent) first. Hybrid still beats dense-only on hand-typed tickets (data export, password reset, laptop boot).
- **σ must be picked on hand-typed tickets** (data audit #5). Dataset tickets have reworded copies in the KB and score ~0.98, much higher than anything a judge will type.
- **Known gap:** some topics have no close precedent (for example a German printer ticket matches "several office devices"). The σ gate and the LLM's out-of-scope flag cover this.

## Required deliverables (from the problem statement)

- [ ] **Triage:** predict type, queue and priority, each with a confidence score
- [ ] **Metrics on the 20% holdout split:** Macro-F1, and recall on high-priority Incidents
- [ ] **Escalation gate:** explicit rules, and the reason for every escalation is shown
- [ ] **Retrieval:** top 3–5 similar past tickets, shown with `ticket_id`, similarity score and answer snippet
- [ ] **Draft reply:** in the customer's language, built only from past answers, citing the `ticket_id`s it used, and never inventing prices, SLAs or legal promises
- [ ] **Live console:** judges paste a ticket and see the predictions, confidence, latency, escalation badge, past tickets and draft
- [ ] **Submission:** public git repo with `requirements.txt` and an evaluation script, plus a README and a 5-minute live demo

**Bonus features (only if there's time):** cross-lingual matching (English past tickets for German input and vice versa), outage burst detection, PII masking.

## Dataset

- Hugging Face: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC-BY-NC-4.0, about 61.8k tickets)
- **Inputs:** `subject` (can be empty, so never crash on it) and `body`
- **Targets:** `type`, `queue`, `priority`
- **Language:** detect it from the text at demo time. Don't read the `language` column.

### ⚠️ Leakage rules (breaking these can get us disqualified)

- Use one fixed split for everything: shuffle with `random_state=42`, first 80% = knowledge base + training data, last 20% = holdout.
- Never index or read the holdout split's `answer`, `type`, `queue` or `priority`, except to score predictions at the very end.
- Fit everything (vectorizers, models, the search index) on the 80% split only.
- Tune calibration and every threshold on the calibration slice (halves A and B), never on the holdout.
- Tags (`tag_1`…`tag_8`) are only used to build the outage keyword list, never as model features.
- Neighbour votes never include the ticket itself.

## Open decisions

| Topic | Options | Owner | Decision |
|---|---|---|---|
| Data source | main CSV only / main + deduplicated 20k | | ✅ main CSV only (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`). The 20k file repeats 8,395 of its tickets with 99.7–99.9% identical labels; the German file is a different domain without `type` / `answer`. See `docs/DATA_PREPROCESSING.md` |
| LLM for drafting | Gemini (check the hackathon benefits page) / other | Chirag | ✅ `gemini-2.5-flash` with thinking off (~1–2 s), fallback `gemini-3.1-flash-lite` (~1 s), one 2 s retry on HTTP 429. Called over REST with the standard library (no SDK). Measured 21 Nov: gemini-3.5/3.8-flash took 24 s+ or timed out, gemini-2.5-flash-lite returns 404 for new keys. 2.5-flash rate-limited after ~15 calls in a few minutes: **don't hammer it while rehearsing**. Override with `GEMINI_MODEL` |
| Embedding model | multilingual-e5 / bge-m3 / TF-IDF | | ✅ `intfloat/multilingual-e5-base` (768 dims, `query: ` prefix on both sides). Probe on the 80% split: e5-base found the original from a translated ticket 60% of the time in the top 5, vs 37% for e5-small. Same-language quality was equal, both separate off-topic queries perfectly, and queries take ~12 ms. bge-m3 was not tested (download kept timing out); optional upgrade if there's time |
| Classifier | embeddings + logistic regression / TF-IDF / neighbour votes / ensemble / fine-tuned transformer | | ✅ ensemble of (0) embeddings + LogReg, (1) TF-IDF + LogReg and (2) neighbour votes, combined by (6) (weighted average of log-probs, weights fit on half A; updated after the data audit: it beat the LogReg combiner by +5 points mean Macro-F1 on new tickets) and calibrated on half B; (5) critical detector replaces the keyword overrides. See [Triage in detail](#triage-in-detail-options-0-1-2-5-6) |
| Retrieval | dense only / hybrid | | ✅ hybrid: BM25 + dense, fused with RRF |
| Vector store | FAISS / Chroma / numpy | | Triage uses **numpy** (exact search, ~ms for 23k rows). ⚠️ Avoid FAISS on macOS: imported after PyTorch has run, it segfaults (GPU) or hangs (CPU) |
| UI | Streamlit / Gradio / other | | |
| What happens after "Escalate" | | | |
| What the AI draft must contain | | | |
| Confidence and similarity thresholds | | | Critical threshold: catch ≥ 90% of high-priority Incidents, tuned on half B. Confidence threshold: tuned on half B. ✅ Similarity threshold: σ = 0.86 for e5-base, between the highest off-topic probe (0.853) and the lowest hand-typed ticket (0.870). See [Retrieval results](#retrieval-results) |
| Team roles (5 people) | | | |
