"""Build split -> train triage (90% of KB) -> validate (10% of KB) -> build FAISS index over KB.

Holdout (test.parquet) is never touched here.
Run: python -m scripts.train
"""
import json
import time

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from sentineldesk import config
from sentineldesk.data import TARGETS, build_split
from sentineldesk.retrieval import build_index
from sentineldesk.triage import TriageModel

kb, _ = build_split()
tr, va = train_test_split(kb, test_size=0.1, random_state=config.SEED, stratify=kb["type"])

t0 = time.time()
model = TriageModel().fit(tr)
print(f"triage trained in {time.time()-t0:.0f}s")

probs = model.predict_proba(va["subject"].tolist(), va["body"].tolist())
val = {}
for t in TARGETS:
    classes, P = probs[t]
    pred = classes[P.argmax(1)]
    val[t] = round(f1_score(va[t], pred, average="macro"), 4)
    print(f"\n=== {t} (validation, macro-F1={val[t]}) ===")
    print(classification_report(va[t], pred, digits=3, zero_division=0))
model.save()

# Save validation predictions for threshold tuning (part of KB split, so allowed).
va = va.copy()
for t in TARGETS:
    classes, P = probs[t]
    va[f"pred_{t}"] = classes[P.argmax(1)]
    va[f"conf_{t}"] = P.max(1)
    if t == "priority":
        va["p_high"] = P[:, list(classes).index("high")]
va.drop(columns=["answer"]).to_parquet(config.ARTIFACT_DIR / "val_preds.parquet", index=False)
json.dump(val, open(config.ARTIFACT_DIR / "val_metrics.json", "w"), indent=2)

t0 = time.time()
build_index(kb)
print(f"FAISS index built over {len(kb)} KB tickets in {time.time()-t0:.0f}s")
