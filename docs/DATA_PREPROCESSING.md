# Data audit and preprocessing

As of 21 Nov 2026 · file audited: `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (the file `split.py` loads as `data/tickets.csv`)

> **Where these numbers come from.** The audit was run in a separate prototype on the same CSV with the same split recipe as `split.py` (shuffle seed 42, 80/20), so ticket ids and counts match this repo exactly: 22,869 knowledge base (KB) / 5,718 holdout. Facts about the data hold for this repo as they are. Scores marked *prototype* come from that prototype's own TF-IDF classifier and multilingual-e5-small embeddings, not from our ensemble: read them as indicative. The prototype's scripts and cleaned files are not in this repo; datasets stay out of git (team rule) and can be shared through Drive like `data/index/`.

## Summary

The data is usable but has four traps: about half the tickets are reworded copies of each other, about 0.5% of customer messages were written in the agent's voice, 27% of `de` language labels are wrong, and queue/priority labels are only partly predictable from the text. Missing values are harmless and need no imputation.

| Finding | Number |
| --- | --- |
| Tickets in the file we use | 28,587 (22,869 KB / 5,718 holdout) |
| Holdout tickets with a reworded copy in the KB (cosine ≥ 0.98) | 2,669 (46.7%); exact duplicates: 0 |
| Calibration tickets with a reworded copy in training, with today's random `calib_split` | 42% |
| KB rows with a broken customer message (agent-written, or a copy of the answer) | 124 (0.5%) |
| KB rows unfit as precedents (adds missing / echo-only answers, very short tickets) | 329 (1.4%) |
| `de` labels that are English text (KB) | 2,679 of 9,814 (27.3%) |
| Missing subjects | 3,838 (13.4%), unrelated to the labels |
| Labels that disagree with their own family of copies | 206 |
| Suspected mislabels for human review | 3,088 |
| Tag spellings (KB) | 1,137 → 1,045 after standardising |

## What this means for our repo

Ordered by priority. Items 1–4 change what our metrics and thresholds mean, so they matter before the final holdout run.

| # | Finding | Affects | Suggested change | Priority |
| --- | --- | --- | --- | --- |
| 1 | The 20k file repeats 8,395 of the main file's tickets with the same labels; the German file is a different domain | `split.py`, README "Open decisions" (blocking) | Close the decision: **main CSV only** (section 1) | High, blocking |
| 2 | 46.7% of holdout tickets are rewordings of a KB ticket; `check_rules.py` only counts exact duplicates, which are 0 | `check_rules.py`, `scripts/eval_triage.py` | Report near-duplicates (cosine ≥ 0.98) in `check_rules.py`; report metrics on all holdout tickets **and** on novel ones (no KB copy) | High |
| 3 | 42% of the random calibration slice has a reworded copy in training, so calibration, ensemble weights and the critical-detector threshold are tuned on easy tickets | `sentineldesk/triage/data.py` (`calib_split`) | Draw the slice and halves A/B by family (Appendix A). In the prototype this made confidence on new tickets about 2× better calibrated | High |
| 4 | Neighbour votes: excluding the ticket itself is not enough, because its reworded copies vote with ~99% identical labels | `sentineldesk/triage/neighbours.py`, `search(exclude_ids=…)` | For half-A tickets, exclude the whole family (Appendix A), or the ensemble learns to over-trust neighbour votes | High |
| 5 | The σ in-domain median (0.954) comes from dataset tickets that have copies in the KB; typed tickets score lower | README RAG plan, σ choice | Add hand-written live-style in-domain probes (en + de) when picking σ. Prototype, e5-small: dataset tickets median 0.975, typed probes median 0.887 | Medium |
| 6 | 4.3% of bodies and 3.5% of answers contain the two characters `\n`; 2.2% / 0.6% contain `<br>` | `sentineldesk/triage/text.py` (`build_text`) | Two `replace` calls in `_clean` (Appendix B) | Medium, quick |
| 7 | 124 KB customer messages are broken (agent-written or a copy of the answer); 137 answers only restate the complaint | Training data, `kb.pkl` `searchable` | Drop broken rows from training; add echo-only answers to `searchable = False` next to the empty / under-40-character rule (section 5) | Medium |
| 8 | 12.4% of answers use `[Name]` / `{tel_num}` placeholders instead of `<name>` / `<tel_num>` | Draft prompt, verify step | Normalise to the `<tag>` form before answers reach the LLM; verify keeps only `<…>` placeholders | Medium |
| 9 | `docs/LANGUAGE_DETECTION.md` counts 8.7% mislabelled across all three files; in the file we use it is 11.7% of rows (27% of `de` labels) | `docs/LANGUAGE_DETECTION.md` | Quote the main-file number; the conclusion (never read the column) is the same | Low |
| 10 | 206 labels disagree with a strict majority of their family | Training labels | Optional: train on family-majority labels (prototype: +0.3 to +0.6 macro-F1) | Low |
| 11 | Tags have spelling variants (`DataProtection` / `Data Protection`, plurals, typos, 5 German tags) | `scripts/mine_keywords.py` | Standardise tags before mining outage keywords so variants count together (section 7) | Low |

**Proposed README row (Open decisions):**

| Topic | Options | Owner | Decision |
| --- | --- | --- | --- |
| Data source | main CSV only / main + deduplicated 20k | | ✅ main CSV only (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`). The 20k file repeats 8,395 of its tickets with 99.7–99.9% identical labels; the German file is a different domain without `type` / `answer`. See `docs/DATA_PREPROCESSING.md` |

## 1. Source files

Use only the main file; the other two add duplicates or a different domain. The Hugging Face repo [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC-BY-NC-4.0) ships three CSV files, 61,765 rows in total.

| File | Rows | What it is | Use? | Why | Cons |
| --- | --- | --- | --- | --- | --- |
| `aa_dataset-tickets-multi-lang-5-2-50-version.csv` | 28,587 | IT/SaaS tickets, 16 columns, en + de | Yes | Recommended by the brief; full schema (subject, body, answer, type, queue, priority, language, 8 tags) | Smaller than all three files combined |
| `dataset-tickets-multi-lang-4-20k.csv` | 20,000 | Older generation of the same tickets | No | 8,395 bodies are identical to the main file, with 99.7–99.9% identical labels: adding it puts the same ticket in training and holdout | Its other ~11,600 rows are unused; they may be near-copies too (not checked) |
| `dataset-tickets-german_normalized_50_5_2.csv` | 13,178 | Separate German dataset | No | Different domain (42 general-topic queues such as Pets or Travel), 5 priority levels, no `type` or `answer` columns | Fewer German examples than are technically available |

The brief's `load_dataset` snippet loads all three files together; teams using it will have different `TKT-xxxxx` ids from ours.

## 2. Split and leakage controls

`split.py` already follows the brief and keeps holdout answers inside the function; the gap is that a random split and a random calibration slice both put reworded copies on both sides.

| Step | In this repo today | Prototype addition | Why | Cons |
| --- | --- | --- | --- | --- |
| 80/20 split | `split.py`: seed 42, `TKT-00000`…, 22,869 / 5,718 | Same split, unchanged | One split for everyone; comparable with the brief | Reworded copies land on both sides (section 8) |
| Holdout answers | Dropped inside `load_split` | Same | The brief forbids indexing them | None |
| Holdout labels | Read only to score at the end | Kept in a separate file, loaded after predictions are frozen | Makes accidental reads impossible | Evaluation has to join files |
| Calibration slice | `calib_split`: random 10%, halves A/B | Drawn by whole families; 0 families shared with training | Calibration and thresholds are tuned on tickets that look new, like live traffic | About 1 point lower holdout F1 in the prototype (those families leave training) |
| Duplicate check | `check_rules.py`: exact duplicates (0 found) | Near-duplicates by embedding cosine ≥ 0.98 | Exact matching misses rewordings | Needs holdout embeddings (text only, allowed) |

## 3. Text normalisation

Fix formatting artefacts on load and bring answer placeholders to one form, so the classifier, the search and the drafting LLM see the same text.

| Step | Affected | What to do | Why | In this repo | Cons |
| --- | --- | --- | --- | --- | --- |
| Literal `\n` escapes | 4.3% of bodies (94% of generator version 51), 3.5% of answers | Turn the two characters `\` + `n` into real line breaks | Otherwise they become fake tokens for TF-IDF and junk in the console | Not handled; `build_text` only applies NFKC (Appendix B) | None |
| `<br>` tags | 2.2% of bodies, 0.6% of answers | Turn into line breaks | Same | Not handled | None |
| Placeholder styles | `<name>` style in 47.0% of answers; `[Name]` / `{name}` style in 12.4% of answers and 1.0% of bodies | Convert `[Name]`, `{tel_num}` etc. to the `<tag>` form | The generator masked personal data in two styles; the verify step expects `<…>` | Not handled | Names are not unified: `[Telefonnummer]` becomes `<telefonnummer>`, not `<tel_num>` |
| Salutation / sign-off in answers | About 30% of answers open with a salutation | Strip "Dear [Name]," and "Best regards…" before reusing an answer | A past answer is pasted into a new reply that has its own greeting | Not handled | Pattern-based: unusual greetings can survive |
| Model input | Every ticket | `subject + "\n" + body`, missing subject left out | One input for training and live tickets | ✅ `build_text` | None |

## 4. Missing values

Nothing needs imputing: missing subjects carry no label signal, missing answers only matter for the search, and empty tag columns are not really missing.

| Column | Missing | Pattern found | What to do | Why | Cons |
| --- | --- | --- | --- | --- | --- |
| `subject` | 3,838 (13.4%) | Same rate for every type (13–15%) and priority (13–14%); depends on generator version and language (version 51: 0%; version 52: 12.6% de, 19.2% en; version 400: 9.2% de, 15.1% en) | Leave empty (`build_text` already does) | Filling it in invents information; in the prototype, tickets without a subject scored no worse (type F1 0.904 vs 0.875) | Live tickets without a subject rely on the body alone |
| `answer` | 7 (0.02%); 6 in the KB | 5 look like ordinary tickets without a resolution; 2 are also broken rows (shifted columns, agent reply in the body) | `searchable = False` | Nothing to show or ground on | None material |
| `tag_2` … `tag_8` | 0% to 98% per column | Structural: tickets have 1–8 tags, filled left to right, never with gaps | Treat as one tag list per ticket | Empty trailing columns are not missing data | None |
| `body` | 0 | – | – | – | – |
| `type`, `queue`, `priority`, `language` | 0 | Present but noisy (sections 6 and 9) | See those sections | – | – |

## 5. Broken rows

124 KB rows (0.5%) should be kept out of training and 329 (1.4%) out of the search. The main error is the generator writing the *customer message* in the agent's voice ("We appreciate your interest…", "Es tut uns leid…").

| Check | How it works | KB rows | Suggested action | Measured accuracy |
| --- | --- | --- | --- | --- |
| Agent opener | Body starts like a reply ("Thank you for…", "Wir entschuldigen…", or is addressed to `[Name]`) | 44 | Out of training and search | 28 of 30 correct (93%), together with the next rule |
| Body copies the answer | Word overlap between body and answer ≥ 0.85 | 51 | Out of training and search | (see above) |
| Agent-voice detector, strict | A classifier trained on the dataset itself (answers = agent writing, bodies = customer writing) scores each body without having seen that ticket; score ≥ 0.99 | 65 | Out of training and search | 11 of 13 correct (85%) |
| Agent-voice detector, loose | Score 0.90–0.99 | 86 | Out of search only | 17 of 25 correct at ≥ 0.90 (68%) |
| Answer only restates the complaint | Word overlap 0.60–0.85 | 137 | `searchable = False` | 17 of 20 are weak answers (85%) |
| Too little text | Under 6 words in subject + body | 39 | `searchable = False` | Not reviewed |
| No answer | Answer empty | 6 | `searchable = False` (already in the RAG plan) | – |

**Why:** a customer message written as an agent reply teaches the classifier the wrong thing, and a broken past ticket would be shown to operators and grounded on. Training exclusions use only the high-precision checks; the looser checks only remove precedents, where a false alarm costs one of about 22,600.

**How accuracy was measured:** 115 randomly sampled rows were read one by one in an AI-assisted review (not by a person). Of 40 random rows that passed every check, 1 was still an agent-voice body, so about 2.5% of broken rows remain. The ticket ids of the reviewed rows and of every flagged row are in the prototype's output files (section 13).

**Cons**

- Samples are small: treat the percentages as ±10–15 points.
- About 15% of strict detector exclusions are genuine customer tickets (roughly 10 rows lost from training).
- About 2.5% of the remaining rows are probably still broken.
- The thresholds (0.85, 0.60, 0.99, 0.90, 6 words) are judgement calls checked on samples.

## 6. Language labels

The `language` column is wrong for 27% of tickets labelled German: they are English text. This confirms the team rule never to read the column; `sentineldesk/language.py` is the detector to use.

| Label ↓ / detected → | German | English |
| --- | --- | --- |
| `de` (KB) | 7,135 | **2,679** |
| `en` (KB) | 2 | 13,053 |

| Item | Detail |
| --- | --- |
| Finding | 2,679 of 9,814 KB tickets labelled `de` are English (27.3%); in all KB rows, 11.7% of labels are wrong |
| Check | On disagreements, `langdetect` sided with the text-based detection 99% of the time |
| Relation to `docs/LANGUAGE_DETECTION.md` | That doc reports 8.7% (5,377 rows) across all three files. The main file alone is noisier (11.7%); same conclusion |
| What to do | Use detected language everywhere: reply language, same-language boost, any per-language analysis |
| Cons | Any metric computed against the `language` column (e.g. "detector accuracy") looks like ~88% when the detector is right |

## 7. Tags

KB tags drop from 1,137 spellings to 1,045 after standardising (92 spellings remapped, 205 tag uses). Tags are only used for the outage keyword list in this repo, so this matters for `scripts/mine_keywords.py`.

| Rule | Example | Why |
| --- | --- | --- |
| Trim whitespace and line breaks | `System\n\n` → `System` | Invisible characters make two tags look different |
| Fold case, spaces, hyphens and CamelCase | `DataProtection` → `Data Protection`; `Decision-Making` → `Decision Making` | Same tag written differently |
| Fold simple plurals | `Campaigns` → `Campaign`; `Protocols` → `Protocol` | Singular and plural mean the same tag |
| Pick the most common spelling as the standard | `Best Practice` → `Best Practices` (39 vs 15 uses) | Keeps the form people used most |
| Map rare typos (≤ 2 uses, ≥ 92% similar to a common tag) | `Feeedback` → `Feedback`; `Documentatoin` → `Documentation` | Typos split counts |
| Translate the five German tags | `Fehlerbehebung` → `Troubleshooting`; `Anleitung` → `Guide` | One vocabulary across languages |

**Cons:** "most common spelling wins" can pick a less natural form (`Fees` over `Fee`); synonyms written with different words are not merged; about half of all tag values are still used only once.

## 8. Reworded copies (paraphrase families)

Half the tickets are rewordings of another ticket with the same labels, and 46.7% of holdout tickets have such a copy in the KB. Holdout scores therefore mix recognising known tickets with handling new ones; report both.

Tickets whose embeddings have cosine ≥ 0.98 are grouped into families (connected components; Appendix A). On all 28,587 tickets with multilingual-e5-small: 19,687 families, 5,543 with more than one member (14,443 tickets, 50.5%); the largest has 46 tickets; no family mixes English and German. Re-check the 0.98 cut-off on e5-base, whose scores are distributed differently: pick the similarity where label agreement drops, as in the table below.

**How often a KB ticket and its nearest other KB ticket share labels (KB labels only)**

| Similarity of nearest ticket | Tickets | Same type | Same queue | Same priority |
| --- | --- | --- | --- | --- |
| 0.99–1.00 (reworded copy) | 5,203 | 100.0% | 99.9% | 99.8% |
| 0.98–0.99 (reworded copy) | 4,903 | 100.0% | 98.5% | 98.4% |
| 0.95–0.98 (related ticket) | 9,370 | 89.2% | 65.1% | 70.6% |
| below 0.95 | 3,393 | 73.5% | 39.6% | 48.2% |

Above 0.98 the labels match ~99% of the time; just below, related tickets agree on queue only two-thirds of the time. Queue and priority are therefore only partly predictable from the text, whatever the model.

| Action | Why | Effect in the prototype | Cons |
| --- | --- | --- | --- |
| Calibration slice drawn by whole families | Calibration and thresholds tuned on tickets with no copy in training, like live tickets | Calibration error on new tickets about halved (type 0.044 → 0.027, queue 0.084 → 0.034, priority 0.087 → 0.045) | Confidence becomes too low on near-copies; more tickets fall under the confidence floors |
| Report all holdout tickets and novel ones | Honest estimate for live traffic | Novel tickets: type 0.835, queue 0.396, priority 0.487 macro-F1 (all: 0.879 / 0.604 / 0.622) | Two sets of numbers to explain |
| Keep the official split unchanged | Comparable with the brief and other teams | – | The official score stays inflated by copies |

## 9. Label noise

Fix the label errors we can be sure of (206 labels that disagree with their own family) and send 3,088 likely errors to a person instead of rewriting them. In the prototype, training on the fixed labels helped a little and dropping suspected errors hurt.

| Check | How it works | Type | Queue | Priority | Suggested action |
| --- | --- | --- | --- | --- | --- |
| Family majority | A copy that disagrees with a strict majority of its family is a slip (copies agree ~99%); ties left alone | 1 | 106 | 99 | Fix in the training labels |
| Confident learning, default | A model trained 5 times on four fifths of the KB (families kept in one fold) predicts the held-out fifth; a label is suspect when another class is clearly more likely | 14.1% | 52.1% | 48.0% | Don't use: flags half of all labels |
| Confident learning, strict | Same, but the given label gets under 2% probability | 271 | 1,862 | 955 | Human review queue |

The default setting flags half of queue and priority labels because they are only partly predictable from text (section 8), not because half are wrong. Samples of the strict set: queue flags are mostly real errors ("Human Resources" for a data-loss analysis), priority flags are mixed ("low" for a medical-records breach is real; medium vs high is often debatable), type flags are mostly debatable Incident vs Problem calls.

**Prototype experiment** (flags recomputed without the validation fold; macro-F1 on validation tickets with no copy in training, original labels):

| Training setup | Rows dropped (type / queue / priority) | Type | Queue | Priority |
| --- | --- | --- | --- | --- |
| A. Original labels | 0 / 0 / 0 | 0.827 | 0.399 | 0.458 |
| **B. Family-majority labels** | 0 / 0 / 0 | **0.830** | **0.405** | **0.458** |
| C. B + drop strict suspects | 245 / 1,669 / 872 | 0.828 | 0.385 | 0.448 |
| D. B + drop suspects under 5% | 555 / 3,539 / 2,073 | 0.836 | 0.382 | 0.458 |

**Conclusion:** use family-majority labels (B) and drop nothing. B is at least as good as A on every label; dropping suspects costs about 2 points of queue F1; D's +0.5 on type is within the noise of one validation fold.

**Cons:** the suspect flags come from the same kind of model, so they favour what it can already predict; only a person can settle the debatable cases (3,088 rows is several hours); gains from B are small; scores against noisy validation labels under-reward cleaning.

## 10. Short-ticket augmentation (prototype)

Typed tickets are much shorter than the dataset's (median 60 words of templated text vs one or two sentences). Training on each ticket twice, in full and as a short version, protects the TF-IDF model against that. Worth trying for our model (1).

| Item | Detail |
| --- | --- |
| What | Short view = greeting line removed ("Dear Support Team,"), then only the first two sentences, subject kept |
| Why | Cutting validation tickets to two sentences dropped queue F1 from 0.61 to 0.51 for a TF-IDF model trained on full text |
| Effect, short tickets | Queue F1 0.513 → 0.591, priority 0.567 → 0.606, type 0.856 → 0.874 |
| Effect, full tickets | No loss: type 0.878 → 0.888, queue 0.610 → 0.605, priority 0.621 → 0.631 |
| Cons | Training takes about twice as long; the short view simulates typed tickets; label errors count twice |

## 11. Holdout handling

The holdout's rows, text and labels stay untouched; it can be flagged and sliced from its text alone, which the team rules allow.

| Step | What | Why | Cons |
| --- | --- | --- | --- |
| No cleaning of rows | All 5,718 holdout tickets stay in the official score | The brief's protocol; removing broken tickets would flatter the score | About 0.6% broken tickets stay in |
| Text-only quality flags | Agent-opener rule, agent-voice detector (trained on the KB only), 6-word rule: 36 tickets (0.6%) | Shows how much broken tickets affect the score, without reading holdout answers or labels | The checks miss some broken rows |
| Copy flags | Nearest-KB similarity per holdout ticket, from text only | Splits the score into known vs new tickets | Needs holdout embeddings |
| Reported slices | All tickets, reworded copies, novel tickets, subject present / missing, clean holdout | Honest view next to the official number | More numbers to explain in the demo |

## 12. Live tickets (inference time)

Live input must go through the same cleaning as the training data before triage and search.

| Step | What it does | In this repo | Note |
| --- | --- | --- | --- |
| Text cleaning | `\n` escapes, `<br>`, whitespace; missing subject allowed | ⚠️ `build_text` lacks the `\n` / `<br>` fix | Appendix B |
| Language detection | en/de from the text | ✅ `sentineldesk/language.py` | Never the `language` column |
| PII masking | Emails, phone numbers, IPs, IBANs, cards, account numbers → `<email>`, `<tel_num>`, … | ✅ `pii.py` | Same placeholder style as the dataset |
| Placeholder normalisation of retrieved answers | `[Name]` → `<name>` before the drafting LLM | ⚠️ not yet | Section 3 |

## 13. Prototype outputs (not in this repo)

Datasets stay out of git (team rule). These files exist in the prototype and can be shared through Drive if useful:

| File | Contents |
| --- | --- |
| `tickets_clean.parquet` (16 MB) | All 28,587 tickets: split, cleaned text, cleaned answer, labels, detected language, clean tags, family id, every flag |
| `label_review_queue.csv` | 3,088 suspected mislabels with the suggested label and an empty `reviewer_decision` column |
| `tag_map.csv` | Raw tag → standard tag |
| `flag_review.csv` | The 115 reviewed rows behind the accuracy figures |
| `DATA_REPORT.md` | Full audit with every number |

## 14. Limitations

| Limitation | Impact | Next step |
| --- | --- | --- |
| Queue and priority labels are partly random | Caps macro-F1 on new tickets (prototype: queue 0.40, priority 0.49) | Work through the review queue, lowest model confidence first |
| The official split contains reworded copies | Holdout score overstates live performance | Always show the novel-ticket slice next to it |
| Broken-row checks reviewed by AI on small samples | Accuracy figures are ±10–15 points | A person spot-checks about 30 flagged rows |
| About 2.5% of rows are probably still broken | Small noise in training and search | Improve the agent-voice detector with the reviewed rows |
| The 0.98 family cut-off was set with e5-small | May need adjusting for e5-base | Re-run the agreement table on `data/index/emb.npy` |
| Placeholder names not unified (`<telefonnummer>` vs `<tel_num>`) | Cosmetic | Map common variants to the dataset's standard names |

## Appendix A: paraphrase families and a family-aware calibration slice

Tested on the KB embeddings (multilingual-e5-small, 22,869 rows): 0 families shared between training and the calibration slice, and 0 between halves A and B. With today's random slice, 42% of calibration tickets have a copy in training.

```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit


def paraphrase_families(emb, threshold=0.98, chunk=2048):
    """One family id per row: tickets whose normalised embeddings have cosine >= threshold."""
    rows, cols = [], []
    for start in range(0, len(emb), chunk):
        r, c = np.nonzero(emb[start:start + chunk] @ emb.T >= threshold)
        keep = r + start < c
        rows.append(r[keep] + start)
        cols.append(c[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(emb), len(emb)))
    return connected_components(graph, directed=False)[1]


emb = np.load("data/index/emb.npy")          # KB rows, same order as kb.pkl / load_split()[0]
family = paraphrase_families(emb)

# 90% train / 5% half A / 5% half B, whole families only
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_idx, calib_idx = next(gss.split(emb, groups=family))
a_pos, b_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
                    .split(calib_idx, groups=family[calib_idx]))
half_a_idx, half_b_idx = calib_idx[a_pos], calib_idx[b_pos]

# Neighbour votes for a half-A ticket i: exclude its whole family, not just itself
# exclude_ids = kb.ticket_id[family == family[i]].tolist()
```

## Appendix B: literal `\n` and `<br>` in `build_text`

Tested on the full raw file: 1,233 bodies and 1,005 answers with literal `\n` escapes go to 0, and no `<br>` is left. A drop-in change to `_clean` in `sentineldesk/triage/text.py`:

```python
def _clean(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).replace("\\n", "\n").replace("<br>", "\n")
```
