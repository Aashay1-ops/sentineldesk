"""Benchmark retrieval models on the KB validation slice (query ticket excluded from its own results).

Relevance proxies (no human judgments exist), labels used only for scoring:
  top1_queue / top1_type : top-1 precedent shares the query's true queue / type
  top5_queue             : share of the top-5 precedents in the query's queue
  tag_jaccard@1          : overlap of the issue tags (tag_1..tag_8) of query and top-1 precedent
  mrr_queue              : 1/rank of the first precedent in the right queue (top 10)

Run: python -m scripts.bench_embeddings [--models minilm e5-small ...]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from sentineldesk import config
from sentineldesk.data import load_kb, ticket_text

MODELS = {
    # key: (hf id, query prefix, passage prefix)
    "minilm":     ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "e5-small":   ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "mpnet":      ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
    "e5-base":    ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "e5-large-instruct": ("intfloat/multilingual-e5-large-instruct",
                          "Instruct: Given a customer support ticket, retrieve similar resolved support tickets\nQuery: ", ""),
    "bge-m3":     ("BAAI/bge-m3", "", ""),
}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=list(MODELS) + ["tfidf", "bm25"])
ap.add_argument("--n", type=int, default=1000)
args = ap.parse_args()

kb = load_kb()
kb = kb[kb["answer"].str.strip() != ""].reset_index(drop=True)
docs = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
tags = kb[[f"tag_{i}" for i in range(1, 9)]].apply(lambda r: {str(t).lower() for t in r if pd.notna(t) and str(t).strip()}, axis=1).tolist()
pos = {t: i for i, t in enumerate(kb["ticket_id"])}

val = pd.read_parquet(config.ARTIFACT_DIR / "val_preds.parquet").sample(args.n, random_state=config.SEED)
val = val[val["ticket_id"].isin(pos)]
qidx = np.array([pos[t] for t in val["ticket_id"]])
queries = [docs[i] for i in qidx]
K_QUEUE, K_TYPE = np.asarray(kb["queue"].tolist(), dtype=object), np.asarray(kb["type"].tolist(), dtype=object)
q_queue, q_type = K_QUEUE[qidx], K_TYPE[qidx]


def score(S: np.ndarray) -> dict:
    """S: (n_queries, n_docs) similarity matrix."""
    S = S.copy()
    S[np.arange(len(qidx)), qidx] = -np.inf  # exclude the query ticket itself
    top = np.argsort(-S, axis=1)[:, :10]
    t1 = top[:, 0]
    jac = [len(tags[q] & tags[d]) / max(1, len(tags[q] | tags[d])) for q, d in zip(qidx, t1)]
    rr = []
    for n, row in enumerate(top):
        hit = np.where(K_QUEUE[row] == q_queue[n])[0]
        rr.append(1 / (hit[0] + 1) if len(hit) else 0)
    return {
        "top1_queue": round(float((K_QUEUE[t1] == q_queue).mean()), 3),
        "top5_queue": round(float((K_QUEUE[top[:, :5]] == q_queue[:, None]).mean()), 3),
        "top1_type": round(float((K_TYPE[t1] == q_type).mean()), 3),
        "tag_jaccard@1": round(float(np.mean(jac)), 3),
        "mrr_queue@10": round(float(np.mean(rr)), 3),
    }


out_path = config.ROOT / "reports" / "embedding_bench.json"
out_path.parent.mkdir(exist_ok=True)
results = json.loads(out_path.read_text()) if out_path.exists() else {}

for key in args.models:
    t0 = time.time()
    if key == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(docs)
        D = vec.transform(docs)
        build = time.time() - t0
        t1 = time.time()
        S = (vec.transform(queries) @ D.T).toarray()
        q_ms = (time.time() - t1) * 1000 / len(queries)
        info = {"params": "-", "dim": "sparse"}
    elif key == "bm25":
        import bm25s
        from sentineldesk.retrieval import tokenize
        bm = bm25s.BM25()
        bm.index([tokenize(d) for d in docs], show_progress=False)
        build = time.time() - t0
        t1 = time.time()
        S = np.zeros((len(queries), len(docs)), dtype="float32")
        ids, sc = bm.retrieve([tokenize(q) for q in queries], k=50, show_progress=False)
        for n in range(len(queries)):
            S[n, ids[n]] = sc[n]
        q_ms = (time.time() - t1) * 1000 / len(queries)
        info = {"params": "-", "dim": "sparse"}
    else:
        from sentence_transformers import SentenceTransformer
        hf, qp, dp = MODELS[key]
        m = SentenceTransformer(hf)
        m.max_seq_length = 256
        cache = config.ARTIFACT_DIR / f"emb_{key}.npy"
        if cache.exists():
            D = np.load(cache)
        else:
            D = m.encode([dp + d for d in docs], batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype("float32")
            np.save(cache, D)
        build = time.time() - t0
        t1 = time.time()
        Q = m.encode([qp + q for q in queries], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        q_ms = (time.time() - t1) * 1000 / len(queries)
        S = Q @ D.T
        info = {"params": f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f}M", "dim": D.shape[1]}
    results[key] = {**score(S), **info, "index_build_s": round(build), "query_ms": round(q_ms, 1)}
    print(key, results[key], flush=True)
    json.dump(results, open(out_path, "w"), indent=2)

print(pd.DataFrame(results).T.sort_values("top1_queue", ascending=False).to_string())
