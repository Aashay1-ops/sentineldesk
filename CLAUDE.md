# SentinelDesk: rules for anyone (human or AI) working in this repo

Read README.md first. It holds the agreed plan. These rules are not up for debate mid-hackathon.

## Hard rules (breaking any of these can disqualify the team)

- **One split, from `split.py` only.** Never re-split, re-shuffle or load the CSV another way to train or index.
- **Holdout = scoring only.** Don't read `test`'s `type`, `queue` or `priority` except to compute metrics at the end. `answer` never leaves `split.py`.
- **Fit on the 80% split only:** classifiers, calibration, BM25/vectorizers, embedding index, thresholds, outage keyword lists.
- **Never tune on the holdout.** Don't pick thresholds, keywords or prompts by looking at holdout errors. Tune on a slice of the 80% split.
- **The live demo uses raw text only.** Detect the language from the text; never read the `language` column at inference.
- **Run `python check_rules.py` before every push.** It must print `OK`.

## Architecture rules (agreed, see README "Architecture (hybrid)")

- Triage = embeddings + LogReg + calibration, plus outage keyword rules. **No LLM in triage.**
- Retrieval = BM25 + dense, fused with RRF. **The UI and gate thresholds use dense cosine**, never the RRF score.
- The LLM only drafts the reply, from retrieved answers only. The verify step must pass, or the ticket escalates.
- Mask PII before any external LLM call.
- Report metrics both "ML only" and "ML + rules".

## Scope rules (avoid drift)

- Build the required MVP (README "Required deliverables") before any bonus feature.
- Bonus features are only: cross-lingual matching, burst detection, PII masking. Nothing else.
- No fine-tuning transformers, no new services or databases, no new dependencies without telling the team.
- If you change a decision, update the README's "Open decisions" table in the same commit.
