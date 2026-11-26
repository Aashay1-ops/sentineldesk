"""② Hybrid precedent retrieval over the KB split only.

dense (multilingual embeddings, FAISS)  ─┐
                                          ├─ Reciprocal Rank Fusion ─ near-duplicate collapse ─ top-k
BM25 (exact terms, codes, versions)     ─┘

RRF only ORDERS results. Every hit carries its dense cosine to the query, which the UI shows
and the "no precedent" gate threshold uses (RRF scores are not similarities).
"""
from __future__ import annotations

import os
import re

import bm25s
import faiss
import numpy as np
import pandas as pd

from . import config
from .data import ticket_text

INDEX_PATH = config.ARTIFACT_DIR / "kb.faiss"
VECS_PATH = config.ARTIFACT_DIR / "kb_vecs.npy"
BM25_DIR = config.ARTIFACT_DIR / "kb_bm25"
META_PATH = config.ARTIFACT_DIR / "kb_meta.parquet"
META_COLS = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]

RRF_K = 60          # standard RRF constant: dampens the gap between rank 1 and rank 2
POOL = 50           # candidates taken from EACH retriever before fusion
DEDUP_COSINE = 0.97  # hits this similar to an already-chosen hit are near-duplicates
# Weighted RRF: BM25's vote relative to dense. 1.0 = plain RRF as in the team README.
# KB validation (n=1000): top-1 same-queue 0.777 @1.0, 0.794 @0.1, 0.816 @0.0 (dense only).
BM25_WEIGHT = float(os.getenv("SENTINEL_BM25_WEIGHT", 1.0))

# ---------------------------------------------------------------- tokenizer (BM25)
# Keeps codes and versions whole: "0x80070005", "php 8.0", "win-11", "e-mail".
_TOKEN_RE = re.compile(r"[a-z0-9äöüß]+(?:[._\-/][a-z0-9äöüß]+)*")
_STOP = set("""
a an and are as at be been but by can could do does for from had has have hello hi how i if in into is it its
me my no not of on or our please regards so that the their them there these this to us was we were what when
which who will with would you your dear team support thank thanks
aber als am an auch auf aus bei bin bis bitte da dank danke das dass dem den der des die dies diese doch du
ein eine einem einen einer es für hallo haben hat ich ihr ihre im in ist ja kann können mit mein meine mich
mir nach nicht noch nur oder sehr sich sie sind so team über um und uns unser unsere von vor war wir wird
wurde zu zum zur sehr geehrte geehrter freundlichen grüßen
""".split())


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]
    return toks or ["<empty>"]


# ---------------------------------------------------------------- embeddings (dense)
_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(config.EMBED_MODEL)
    return _encoder


def embed(texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
    v = encoder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=show_progress, convert_to_numpy=True)
    return v.astype("float32")


# ---------------------------------------------------------------- build (offline, KB only)
def build_index(kb: pd.DataFrame, vecs: np.ndarray | None = None):
    """Index ONLY the KB split. Holdout tickets and answers never enter either index."""
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    kb = kb[kb["answer"].str.strip() != ""].reset_index(drop=True)
    texts = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]

    if vecs is None:
        vecs = embed(texts, show_progress=True)
    np.save(VECS_PATH, vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    bm25 = bm25s.BM25()                         # IDF statistics come from the KB only
    bm25.index([tokenize(t) for t in texts], show_progress=False)
    bm25.save(str(BM25_DIR))

    kb[META_COLS].to_parquet(META_PATH, index=False)


# ---------------------------------------------------------------- search (online)
class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.vecs = np.load(VECS_PATH)
        self.bm25 = bm25s.BM25.load(str(BM25_DIR))
        self.meta = pd.read_parquet(META_PATH)
        self.kb_ids = set(self.meta["ticket_id"])
        self._ids = self.meta["ticket_id"].to_numpy()

    # --- the two retrievers, each returning candidate row indices in rank order
    def _dense(self, qvec: np.ndarray, n: int) -> list[int]:
        _, idx = self.index.search(qvec.reshape(1, -1), n)
        return [int(i) for i in idx[0] if i >= 0]

    def _bm25(self, text: str, n: int) -> list[int]:
        docs, scores = self.bm25.retrieve([tokenize(text)], k=n, show_progress=False)
        return [int(d) for d, s in zip(docs[0], scores[0]) if s > 0]

    def search(self, subject: str, body: str, k: int = config.TOP_K, mode: str = "hybrid",
               exclude_ids: set | None = None) -> tuple[list[dict], np.ndarray]:
        text = ticket_text(subject, body)
        qvec = embed([text])[0]
        return self.search_with_vec(text, qvec, k, mode, exclude_ids), qvec

    def search_with_vec(self, text: str, qvec: np.ndarray, k: int = config.TOP_K, mode: str = "hybrid",
                        exclude_ids: set | None = None, bm25_weight: float | None = None) -> list[dict]:
        dense = self._dense(qvec, POOL) if mode in ("dense", "hybrid") else []
        sparse = self._bm25(text, POOL) if mode in ("bm25", "hybrid") else []

        # Reciprocal Rank Fusion: score = Σ 1 / (RRF_K + rank) over the retrievers that found the doc
        fused: dict[int, float] = {}
        rank_d = {i: r for r, i in enumerate(dense, 1)}
        rank_b = {i: r for r, i in enumerate(sparse, 1)}
        w_b = BM25_WEIGHT if bm25_weight is None else bm25_weight
        for i in set(rank_d) | set(rank_b):
            fused[i] = (1.0 / (RRF_K + rank_d[i]) if i in rank_d else 0.0) + \
                       (w_b / (RRF_K + rank_b[i]) if i in rank_b else 0.0)
        order = sorted(fused, key=fused.get, reverse=True)

        # Walk the fused list: skip excluded IDs and near-duplicates of hits already chosen.
        chosen: list[int] = []
        for i in order:
            if exclude_ids and self._ids[i] in exclude_ids:
                continue
            if any(float(self.vecs[i] @ self.vecs[j]) >= DEDUP_COSINE for j in chosen):
                continue
            chosen.append(i)
            if len(chosen) == k:
                break

        hits = []
        for i in chosen:
            row = self.meta.iloc[i].to_dict()
            row["similarity"] = float(self.vecs[i] @ qvec)   # dense cosine: shown in UI, used by the gate
            row["rrf_score"] = fused[i]
            row["dense_rank"] = rank_d.get(i)
            row["bm25_rank"] = rank_b.get(i)
            row["found_by"] = "both" if i in rank_d and i in rank_b else ("dense" if i in rank_d else "bm25")
            hits.append(row)
        return hits

    # kept for scripts that only need dense neighbours
    def search_vec(self, qvec: np.ndarray, k: int = config.TOP_K) -> list[dict]:
        return self.search_with_vec("", qvec, k, mode="dense")
