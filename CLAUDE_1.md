# CLAUDE.md — SentinelDesk triage module

This file tells Claude Code how to build the **triage** part of SentinelDesk (step ① in `README.md`). Read it fully before writing code. If anything here conflicts with `README.md`, this file wins for triage; ask before changing anything outside triage.

## Project context (short)

SentinelDesk is a 6-hour hackathon project. An incoming support ticket (`subject` may be empty, `body`, English or German) is:

1. **triaged**: type, queue, priority with calibrated confidence, plus a separate "critical" score (this module),
2. matched to similar past tickets (retrieval, another teammate),
3. passed through an escalation gate (another teammate),
4. answered by a grounded LLM draft if safe (another teammate).

**Goal of this module:** predict `type` (Incident / Request / Problem / Change), `queue` (about 10 departments) and `priority` (high / medium / low), each with a trustworthy confidence, and a score for "is this a high-priority Incident?" that the gate uses to escalate outages.

Judges score: **Macro-F1** on the 20% holdout for each task, and **recall on high-priority Incidents**. Any data leakage can disqualify the team.

## The approach (do not change without asking)

Five pieces. Numbers match the team's README.

| # | Piece | What it does |
|---|---|---|
| 0 | Embeddings + logistic regression | One LR per task on multilingual sentence embeddings (meaning, works across en/de) |
| 1 | TF-IDF + logistic regression | One LR per task on word + character n-grams (exact words, product names, error codes) |
| 2 | Neighbour votes | Labels of the top-20 most similar training tickets, weighted by cosine similarity. No training |
| 6 | Ensemble + calibration | Per task, a small LR combines the probabilities from 0, 1 and 2; then temperature scaling |
| 5 | Critical detector | Binary LR on embeddings + outage-keyword features → P(high-priority Incident); threshold tuned for recall ≥ 0.90 |

Rules of the approach:

- **Keyword matches never overwrite predicted labels.** They are input features to the critical detector only.
- **No LLM calls anywhere in triage.**
- **Fallback mode:** if ensemble artifacts are missing, predict with model 0 alone, calibrated with temperature scaling on half B, plus the critical detector. Output `"mode": "fallback"`.

## Data

### Source

- Hugging Face dataset `Tobi-Bueck/customer-support-tickets`, file **`aa_dataset-tickets-multi-lang-5-2-50-version.csv`**.
- Do **not** call `load_dataset("Tobi-Bueck/customer-support-tickets")` without `data_files`: it merges several versions (dozens of queues, 5 priorities).
- Find the exact path with `huggingface_hub.list_repo_files(..., repo_type="dataset")` and download with `hf_hub_download`. Fail loudly with a clear message if the file isn't found.
- The 80/20 split is the team's `split.load_split("data/tickets.csv")` (repo root). Triage never re-splits; `data.py` downloads the agreed file to `data/tickets.csv` only if it is missing (config `data.csv_path`), and cuts half A / half B out of kb.
- After loading, print and sanity-check: row count, unique values of `type`, `queue`, `priority`, missing-subject count. Expect 3 priorities and 4 types; stop with an error otherwise.

### Text

```python
def build_text(subject, body) -> str:
    s = "" if subject is None or (isinstance(subject, float) and math.isnan(subject)) else str(subject)
    b = "" if body is None or (isinstance(body, float) and math.isnan(body)) else str(body)
    text = (s.strip() + "\n" + b.strip()).strip()
    if not text:
        raise ValueError("Ticket has no subject and no body")
    return text
```

Never produce the string `"nan"`. Apply Unicode NFKC normalization. Use this one function everywhere: training, evaluation and the live demo.

### Split (must match the team exactly)

```python
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
split_idx = int(len(df) * 0.8)
kb = df.iloc[:split_idx]      # 80%: training + knowledge base
test = df.iloc[split_idx:]    # 20%: holdout
```

Inside `kb`:

- `calib = kb.sample(frac=0.10, random_state=42)`; `train = kb.drop(calib.index)` (≈ 90% of kb).
- Split `calib` into **half A** and **half B**: `half_a = calib.sample(frac=0.5, random_state=42)`; `half_b = calib.drop(half_a.index)`.

| Part | Used for |
|---|---|
| `train` | Fit TF-IDF, model 0, model 1, critical detector; neighbour index |
| `half_a` | Train the ensemble (6) |
| `half_b` | Calibrate (temperature scaling), tune critical threshold, suggest confidence thresholds |
| `test` | Final evaluation only |

`scripts/build_split.py` writes:

- `artifacts/splits/train.parquet`, `half_a.parquet`, `half_b.parquet` (all columns needed for training; `answer` may be kept for the retrieval teammate)
- `artifacts/splits/test_inputs.parquet`: **only** `ticket_id`, `subject`, `body`
- `artifacts/splits/test_labels.parquet`: **only** `ticket_id`, `type`, `queue`, `priority`
- Never write the holdout `answer` column anywhere.

### Leakage rules (hard requirements)

- Nothing reads `test_labels.parquet` except `scripts/eval_triage.py`, and only **after** all predictions are computed and saved.
- Fit every vectorizer, model, index and calibrator on `train`, `half_a` or `half_b` only.
- Never use the `language` column or `tag_1`…`tag_8` as model features.
- Tags may be used **only** to mine candidate outage keywords, from `train` only.
- Never tune keywords, thresholds or hyperparameters by looking at holdout results.
- Add an assertion in training and evaluation that no `test` `ticket_id` appears in any training part or index.

## Components: implementation details

### Shared

- **Label order:** for each task, the sorted unique labels in `train`. Save to `artifacts/triage/labels.json`. Every model returns probabilities as a NumPy array in exactly this order, shape `(n, n_classes)`, rows summing to 1.
- **Seeds:** `random_state=42` everywhere.
- **Config:** `configs/triage.yaml`. Everything tunable lives there (model names, k, thresholds, paths).
- **Sample mode:** every script takes `--sample N` to run on N training tickets (and proportionally smaller other parts), for quick checks.

### Embeddings (shared with retrieval and neighbour votes)

- `sentence-transformers`. Default model `intfloat/multilingual-e5-small` (CPU-friendly); option `BAAI/bge-m3` (use a GPU).
- e5 models need a prefix. Use `"query: " + text` for every ticket. bge-m3 needs no prefix. Put the prefix in config per model.
- `normalize_embeddings=True`, `max_seq_length=512`, batch size from config. Use `device="cuda"` if available, else CPU. Use fp16 on GPU.
- `scripts/embed.py` embeds `train`, `half_a`, `half_b` and `test_inputs` once and saves `artifacts/embeddings/<model_slug>/<part>.npy` plus a `<part>_ids.json` with the ticket order.
- At prediction time, a single ticket is embedded with the same model and prefix.

### Model 0: embeddings + LR

- `LogisticRegression(max_iter=2000, class_weight="balanced", C=<config>)`, one per task, trained on `train` embeddings.
- No calibration at this stage (the ensemble calibrates). In fallback mode, apply temperature scaling fitted on `half_b`.

### Model 1: TF-IDF + LR

- `FeatureUnion` of:
  - `TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=200_000)`
  - `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True, max_features=200_000)`
- Lowercase. Fit on `train` only.
- LR per task with `class_weight="balanced"`, solver `"saga"` (or `"liblinear"` if faster on the sample run), must support `predict_proba`.
- Config option to drop the character n-grams if training is too slow.

### Model 2: neighbour votes

- FAISS `IndexFlatIP` built over **`train`** embeddings only (normalized, so inner product = cosine).
- For a query: top `k` (default 20). Score per class = sum of `max(sim, 0) ** p` over neighbours with that label (default `p = 1`), plus smoothing `alpha` (default 0.1) per class. Normalize to probabilities.
- **Self-exclusion:** if a query `ticket_id` is in the index, drop it from its own neighbour list. Add a test for this.
- Also return, for the UI: the top neighbours (`ticket_id`, similarity, their labels) and an explanation string per task, e.g. `"8 of 20 similar tickets went to Billing and Payments"`.
- `half_a`, `half_b`, `test` and live tickets all search the same `train` index.

### Model 6: ensemble + calibration

- For each task, input features = `log(clip(p, 1e-6, 1))` from models 0, 1 and 2, concatenated (`3 × n_classes` columns).
- `LogisticRegression(max_iter=2000, C=<config>)` trained on **`half_a`** only.
- Calibration: **temperature scaling** fitted on **`half_b`**. Apply to the ensemble's log-probabilities: `p_T = softmax(log p / T)`. Find `T` by minimizing negative log-likelihood with `scipy.optimize.minimize_scalar` (bounds 0.05–20). Implement this once in `calibration.py` and reuse it for fallback mode.
- Do not use `CalibratedClassifierCV(cv="prefit")` (deprecated in recent scikit-learn).
- Save, per task, `T` and the expected calibration error (ECE, 15 bins) on `half_b` before and after scaling.
- **Suggested confidence threshold** per task (for the gate): the confidence below which accuracy on `half_b` drops under 50%, computed in bins. Save to `artifacts/triage/thresholds.json`. The gate owner decides whether to use it.

### Model 5: critical detector

- Target: `critical = (type == "Incident") & (priority == "high")`.
- Features: the embedding, plus keyword features: `any_hit` (0/1), `n_hits` (count), and `n_distinct_hits`. Scale keyword features with a `StandardScaler` fitted on `train`.
- `LogisticRegression(max_iter=2000, class_weight="balanced")` trained on `train`.
- **Threshold:** on `half_b`, choose the highest threshold that still gives recall ≥ `critical.target_recall` (default 0.90). Save threshold, half-B recall and half-B precision to `thresholds.json`.
- Output: `score` (probability), `threshold`, `fires` (score ≥ threshold), and the matched keywords.

### Outage keywords

- Seed list in `keywords/outage_keywords.yaml`, with separate `en` and `de` lists. Seed examples:
  - en: outage, down, not working, cannot access, can't access, no access, unavailable, all users, everyone affected, data loss, breach, ransomware, crashed, telemedicine, EMR
  - de: Ausfall, ausgefallen, kein Zugriff, funktioniert nicht, nicht erreichbar, Störung, alle Mitarbeiter, alle Nutzer, Datenverlust, Sicherheitsvorfall, abgestürzt
- Matching: NFKC-normalize and lowercase the text, then match with word boundaries (Unicode-aware regex). "down" must not match "download"; multi-word phrases must match as phrases.
- `scripts/mine_keywords.py` (optional helper): from `train` only, find words and bigrams much more frequent in tickets tagged `Outage` or `Disruption` than in other tickets. First check those tag values exist in the tag columns and print what is found. Write candidates to `keywords/candidates.txt` for a human to review. Never add them to the YAML automatically.

## Public interface (other teammates depend on this)

`sentineldesk/triage/predict.py`:

```python
class Triage:
    @classmethod
    def load(cls, artifacts_dir: str = "artifacts") -> "Triage": ...
    def predict(self, subject: str | None, body: str) -> dict: ...
    def predict_batch(self, subjects: list[str | None], bodies: list[str]) -> list[dict]: ...
```

- `load` reads all artifacts once (the Streamlit app wraps it in `st.cache_resource`).
- `predict` must work for an empty or missing subject. An empty body plus empty subject raises `ValueError`.
- `predict_batch` embeds in batches. The evaluation script uses it.

Return value of `predict` (exact keys; other modules rely on them):

```json
{
  "mode": "full",
  "type":     {"label": "Incident", "confidence": 0.81, "probs": {"Change": 0.03, "Incident": 0.81, "Problem": 0.06, "Request": 0.10}},
  "queue":    {"label": "Technical Support", "confidence": 0.58, "probs": {"...": 0.0}},
  "priority": {"label": "high", "confidence": 0.74, "probs": {"high": 0.74, "low": 0.06, "medium": 0.20}},
  "critical": {"score": 0.62, "threshold": 0.35, "fires": true, "keyword_hits": ["outage", "emr"]},
  "neighbours": {
    "top": [{"ticket_id": "TKT-01234", "similarity": 0.83, "type": "Incident", "queue": "Technical Support", "priority": "high"}],
    "explanation": {"type": "15 of 20 similar tickets were Incident", "queue": "9 of 20 similar tickets went to Technical Support", "priority": "11 of 20 similar tickets were high"}
  },
  "suggested_conf_thresholds": {"type": 0.45, "queue": 0.40, "priority": 0.50},
  "latency_ms": {"embed": 35.2, "tfidf": 3.1, "models": 4.0, "neighbours": 2.5, "ensemble": 0.4, "critical": 0.3, "total": 46.1}
}
```

- `neighbours.top` lists the top 5 (for display); votes use all `k`.
- In fallback mode, `"mode": "fallback"` and probabilities come from model 0 with temperature scaling.
- Time each stage with `time.perf_counter()`.

## Repo layout

```
sentineldesk/
├── configs/triage.yaml
├── keywords/
│   ├── outage_keywords.yaml
│   └── candidates.txt          # generated, for human review
├── sentineldesk/
│   └── triage/
│       ├── __init__.py
│       ├── config.py           # loads triage.yaml
│       ├── data.py             # download, split, parquet IO, leakage asserts
│       ├── text.py             # build_text, normalization
│       ├── keywords.py         # load list, match, features
│       ├── embed.py            # embedding model wrapper
│       ├── model_embed.py      # model 0
│       ├── model_tfidf.py      # model 1
│       ├── neighbours.py       # model 2 (FAISS)
│       ├── ensemble.py         # model 6
│       ├── calibration.py      # temperature scaling, ECE, reliability data
│       ├── critical.py         # model 5 + threshold tuning
│       └── predict.py          # Triage class
├── scripts/
│   ├── build_split.py
│   ├── embed.py
│   ├── train_triage.py         # trains 0, 1, 5; builds index; trains 6; calibrates; tunes thresholds
│   ├── eval_triage.py
│   └── mine_keywords.py
├── tests/
│   └── test_triage.py
├── artifacts/                  # generated; not committed (add to .gitignore)
├── results/                    # metrics JSON and plots; committed at the end
└── requirements.txt
```

Use `pathlib`, type hints and short docstrings. Use `logging`, not `print`, inside the package. Save models with `joblib`. Keep artifacts paths configurable so Colab can write to Google Drive.

## Evaluation (`scripts/eval_triage.py`)

1. Load `test_inputs.parquet`, run `predict_batch`, save predictions to `results/test_predictions.parquet`.
2. Only then load `test_labels.parquet` and join on `ticket_id`.
3. Report, for **each task**:
   - Macro-F1 for: model 0 alone, model 1 alone, model 2 alone, fallback (0 + calibration) and full ensemble,
   - per-class precision/recall/F1 for the full ensemble,
   - confusion matrix PNG for priority (and queue).
4. High-priority Incident recall and precision:
   - "labels only": predicted type == Incident and predicted priority == high,
   - "critical detector": `fires == True`,
   - plus a precision–recall curve PNG for the detector with the chosen threshold marked.
5. Calibration: ECE (15 bins) before and after temperature scaling on the holdout, plus a reliability diagram PNG per task.
6. Latency: p50 and p95 of `predict` for single tickets, on 200 holdout tickets.
7. Write everything to `results/triage_metrics.json` and PNGs to `results/plots/`.

Never feed holdout results back into training or thresholds.

## Tests (`tests/test_triage.py`, pytest)

- `build_text` handles `None`, `NaN` and empty subjects, never returns `"nan"`, and raises on empty subject + body.
- Split is reproducible: same `ticket_id`s on two runs; train, half A, half B and test are disjoint and cover all rows.
- `test_inputs.parquet` has only `ticket_id`, `subject`, `body`.
- No holdout `ticket_id` in the FAISS index or any training part.
- A ticket never appears in its own neighbour list.
- Keyword matching: "download" does not match "down"; "kein Zugriff" matches as a phrase; German umlauts work.
- `predict` returns all required keys; each `probs` sums to 1 (±1e-6) and follows `labels.json` order; works for an English ticket, a German ticket and an empty subject.
- Fallback mode works when ensemble artifacts are removed.
- Temperature scaling lowers or keeps NLL on half B.

Use `--sample`-sized artifacts or small fixtures so tests run in under a minute.

## Build order (work milestone by milestone)

After each milestone: run the tests, run the scripts with `--sample 2000`, and report timings. Don't start the next milestone until the current one runs end to end.

1. **M1 Data:** config, `text.py`, `data.py`, `build_split.py`, leakage asserts, split tests.
2. **M2 Embeddings:** `embed.py` + `scripts/embed.py`, saved `.npy` files.
3. **M3 Fallback triage (usable on its own):** model 0, temperature scaling, keywords, critical detector + threshold, `Triage.predict` in fallback mode.
4. **M4 TF-IDF:** model 1.
5. **M5 Neighbour votes:** FAISS index, votes, self-exclusion, explanations.
6. **M6 Ensemble:** model 6 on half A, calibration on half B, suggested thresholds, `mode: "full"`.
7. **M7 Evaluation:** `eval_triage.py`, metrics JSON and plots.

If time runs short, M3 is the minimum deliverable; M4–M6 improve it.

## Commands

```bash
pip install -r requirements.txt
python scripts/build_split.py
python scripts/embed.py --model intfloat/multilingual-e5-small
python scripts/train_triage.py
python scripts/eval_triage.py
pytest -q
```

`requirements.txt` for this module: `pandas`, `numpy`, `scikit-learn`, `scipy`, `sentence-transformers`, `torch`, `faiss-cpu`, `huggingface_hub`, `pyarrow`, `joblib`, `pyyaml`, `matplotlib`, `pytest`. Pin versions once everything runs.

## Expected training time (for planning)

About 23k training tickets. Free Colab T4: total ≈ 10–20 min with e5-small, ≈ 15–30 min with bge-m3, mostly embedding and TF-IDF. On CPU only, bge-m3 embedding can take 40–90 min, so use e5-small or a GPU. Log the time of each step.

## Do not

- Read holdout labels or answers before predictions are saved.
- Use `language` or tag columns as features.
- Overwrite predicted labels with keyword rules.
- Call any LLM or external API in triage.
- Use `CalibratedClassifierCV(cv="prefit")`.
- Change the split code, label names or output keys without telling the team.
- Add heavy frameworks (LangChain, vector DB servers). FAISS in-process is enough.
