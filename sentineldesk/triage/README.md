# Triage module (step ①)

Predicts `type`, `queue` and `priority` with calibrated confidence, plus a separate **critical score** (how likely the ticket is a high-priority Incident). The gate uses that score to escalate outages. No LLM is involved. The full spec is in `CLAUDE_1.md`.

## For teammates: how to use it

```python
from sentineldesk.triage.predict import Triage

triage = Triage.load("artifacts")          # Streamlit: wrap in @st.cache_resource
out = triage.predict(subject, body)        # subject may be None or ""
```

Each call returns:

| Key | Content | Used by |
|---|---|---|
| `type` / `queue` / `priority` | `label`, `confidence` and `probs` (in the fixed order from `artifacts/triage/labels.json`) | gate (low confidence), UI |
| `critical` | `score`, `threshold`, `fires`, `keyword_hits` | gate ("escalate to on-call") |
| `suggested_conf_thresholds` | one value per task, tuned on half B | gate owner decides whether to use them |
| `neighbours` | top 5 `train` tickets (id, cosine, labels) and the text "x of 20 similar tickets…" | UI explanation |
| `latency_ms` | time per stage | UI |
| `mode` | `"full"` (ensemble) or `"fallback"` (model 0 + temperature) | UI |

Keyword hits **never** change the labels. They are only inputs to the critical detector.

**For retrieval:** a full `scripts/embed.py` run with the team model (e5-base) writes the shared index from the README spec: `data/index/emb.npy` (22,869 × 768, normalized float32, in `load_split()` order), `kb.pkl` (same order, with `searchable`) and `meta.json` (model + `"query: "` prefix). Triage's own `train`/`half_a`/`half_b` arrays are slices of the same matrix, and live tickets are embedded from the same text as split.py's `text` column, so triage and retrieval can share one embedding per query. The holdout is embedded separately and never enters the index.

## Train

```bash
python scripts/build_split.py          # split.load_split (team split); downloads data/tickets.csv if missing
python scripts/embed.py                # team model e5-base; also writes the shared data/index/ (emb.npy, kb.pkl)
python scripts/train_triage.py         # add --fallback-only for model 0 + critical only
pytest -q tests/test_triage.py
python scripts/eval_triage.py          # holdout: ONCE, at the very end
```

Add `--sample 2000` to any script for a quick run. Sample runs go to `artifacts/sample_2000/` and never overwrite the full artifacts.

**On Colab (from VS Code):** use `notebooks/triage_colab.ipynb`. It clones the repo, trains on a T4 and sends the artifacts back as a zip. On a Mac with Apple silicon, the full run also works locally: embedding uses the GPU (MPS).

## Where each split part goes

`build_split.py` writes the parts, then `embed.py` (team model only) **redraws train / half A / half B by paraphrase family**: tickets with embedding cosine ≥ 0.98 are reworded copies and always land in the same part (`docs/DATA_PREPROCESSING.md`, items 3–4). It also prints the label-agreement table and refuses a threshold that is too low.

| Part | Size | Used for |
|---|---|---|
| `train` | ~20.5k | TF-IDF, model 0, model 1, critical detector, neighbour index |
| `half_a` | ~1.2k | ensemble (6): 3 weights per task, weighted average of log-probabilities |
| `half_b` | ~1.2k | temperature scaling, critical threshold (recall ≥ 0.90), suggested confidence thresholds. No reworded copy in `train`, so these numbers are what to expect on **new** tickets |
| `test` | ~5.7k | `eval_triage.py` only. Labels are read only after predictions are saved. Reported for all / known (reworded copy in kb) / novel tickets |

## Changes made after the data audit

Measured on the family-based half B. Differences under 0.01 are noise for 1.2k tickets.

| Change | Result | Kept? |
|---|---|---|
| Halves A/B drawn by family | Honest numbers for new tickets (queue ~0.36, priority ~0.46, vs 0.53 / 0.63 on a random half B) | ✅ |
| Weighted average instead of the LogReg combiner | **+5 points** mean Macro-F1 (queue +7, priority +5); the LogReg combiner scored below TF-IDF alone | ✅ |
| `\n` / `<br>` cleaning | ±0.002 | ✅ done team-wide in `split.py` + `build_text` (Chirag, PR #6), so embeddings, TF-IDF and live tickets all see clean text |
| Family-majority labels | ±0.006 | ❌ no measurable gain |
| Drop 57 broken training rows | ±0.004 | ❌ no measurable gain |
| Short-ticket augmentation for TF-IDF | +0.003 on full tickets, −0.005 on short | ❌ no gain, and doubles TF-IDF training time |
| FAISS → NumPy neighbour search | Same results. Fixes a segfault/hang on macOS when FAISS loads after PyTorch | ✅ |
