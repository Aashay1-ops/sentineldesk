"""Compare dense vs BM25 vs hybrid (RRF) retrieval.

Relevance proxy (no human judgments exist): a retrieved precedent is "on topic" if it shares the query's
true queue / type. Labels are used ONLY for scoring.

  --split val      KB validation slice (the query ticket itself is excluded from results)  <- tune here
  --split holdout  20% holdout (inputs only for retrieval; labels for scoring)            <- report here

Run: python -m scripts.compare_retrieval --split val --n 1000
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from sentineldesk import config
from sentineldesk.data import load_test_inputs_only, load_test_labels, ticket_text
from sentineldesk.pipeline import detect_language
from sentineldesk.retrieval import Retriever, embed

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "holdout"], default="val")
ap.add_argument("--n", type=int, default=1000)
args = ap.parse_args()

if args.split == "val":
    Q = pd.read_parquet(config.ARTIFACT_DIR / "val_preds.parquet").sample(args.n, random_state=config.SEED)
    labels = Q.set_index("ticket_id")[["queue", "type"]]
else:
    Q = load_test_inputs_only().sample(args.n, random_state=config.SEED)
    labels = load_test_labels().set_index("ticket_id").loc[Q["ticket_id"], ["queue", "type"]]

texts = [ticket_text(s, b) for s, b in zip(Q["subject"], Q["body"])]
vecs = embed(texts)
qlang = [detect_language(t)[0] for t in texts]
ret = Retriever()

results = {}
for mode in ("dense", "bm25", "hybrid"):
    top1_q, at5_q, top1_t, best_cos, cross, lat, gate_low = [], [], [], [], [], [], []
    for tid, text, v, lang in zip(Q["ticket_id"], texts, vecs, qlang):
        t0 = time.perf_counter()
        hits = ret.search_with_vec(text, v, k=5, mode=mode, exclude_ids={tid})
        lat.append((time.perf_counter() - t0) * 1000)
        truth = labels.loc[tid]
        top1_q.append(hits[0]["queue"] == truth["queue"])
        top1_t.append(hits[0]["type"] == truth["type"])
        at5_q.append(np.mean([h["queue"] == truth["queue"] for h in hits]))
        cos = max(h["similarity"] for h in hits)
        best_cos.append(cos)
        gate_low.append(cos < config.GATE["min_top_similarity"])
        cross.append(np.mean([h["language"] != lang for h in hits]))
    results[mode] = {
        "top1_same_queue": round(float(np.mean(top1_q)), 3),
        "top5_same_queue_rate": round(float(np.mean(at5_q)), 3),
        "top1_same_type": round(float(np.mean(top1_t)), 3),
        "best_dense_cosine_mean": round(float(np.mean(best_cos)), 3),
        "no_precedent_rate": round(float(np.mean(gate_low)), 3),
        "cross_lingual_hit_share": round(float(np.mean(cross)), 3),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
    }

df = pd.DataFrame(results).T
print(f"\nsplit={args.split}  n={len(Q)}\n")
print(df.to_string())
out = config.ROOT / "reports"
out.mkdir(exist_ok=True)
json.dump(results, open(out / f"retrieval_{args.split}.json", "w"), indent=2)
