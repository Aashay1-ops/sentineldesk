"""Public triage interface used by the gate, the UI and the evaluation script.

    from sentineldesk.triage.predict import Triage
    triage = Triage.load("artifacts")           # wrap in st.cache_resource in Streamlit
    out = triage.predict(subject, body)         # dict, see CLAUDE.md "Return value of predict"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from sentineldesk.triage import TASKS
from sentineldesk.triage.calibration import apply_temperature
from sentineldesk.triage.config import REPO_ROOT
from sentineldesk.triage.labels import load_labels
from sentineldesk.triage.neighbours import NeighbourIndex
from sentineldesk.triage.text import build_texts

log = logging.getLogger(__name__)
FULL_MODE_FILES = ("model_tfidf.joblib", "neighbours.npz", "ensemble.joblib")


def _resolve(artifacts_dir: str | Path) -> Path:
    p = Path(artifacts_dir)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    return p / "triage"


class Triage:
    def __init__(self, tdir: Path, embedder=None):
        self.dir = tdir
        self.labels = load_labels(tdir / "labels.json")
        self.meta = json.loads((tdir / "meta.json").read_text())
        self.calib = json.loads((tdir / "calibration.json").read_text())
        self.thresholds = json.loads((tdir / "thresholds.json").read_text())
        self.m0 = joblib.load(tdir / "model_embed.joblib")
        self.crit = joblib.load(tdir / "critical.joblib")
        self.kw = joblib.load(tdir / "keywords.joblib")
        self.mode = "full" if all((tdir / f).exists() for f in FULL_MODE_FILES) and self.calib.get("full") else "fallback"
        self.m1 = self.nn = self.ens = None
        nc = self.meta["neighbours"]
        if (tdir / "neighbours.npz").exists():
            z = np.load(tdir / "neighbours.npz")
            self.nn = NeighbourIndex(z["emb"], z["ids"].tolist(), {t: z[f"codes_{t}"] for t in TASKS},
                                     self.labels, nc["k"], nc["p"], nc["alpha"])
        if self.mode == "full":
            self.m1 = joblib.load(tdir / "model_tfidf.joblib")
            self.ens = joblib.load(tdir / "ensemble.joblib")
        if embedder is None:
            from sentineldesk.triage.embed import Embedder

            embedder = Embedder(self.meta["embedding_model"], self.meta["prefix"],
                                self.meta.get("batch_size", 64), self.meta.get("max_seq_length", 512))
        self.embedder = embedder
        log.info("Triage loaded from %s (mode=%s)", tdir, self.mode)

    @classmethod
    def load(cls, artifacts_dir: str | Path = "artifacts", embedder=None) -> "Triage":
        """Load all artifacts once. `embedder` is only injected by tests."""
        return cls(_resolve(artifacts_dir), embedder)

    # ---- core -------------------------------------------------------------------------------
    def predict_arrays(self, texts: list[str], ticket_ids: list[str] | None = None) -> dict:
        """All intermediate probabilities as arrays (used by predict_batch and eval_triage.py)."""
        lat: dict[str, float] = {}
        t0 = time.perf_counter()
        emb = self.embedder.encode(texts)
        lat["embed"] = time.perf_counter() - t0

        t = time.perf_counter()
        p0 = self.m0.predict_proba(emb, self.labels)
        lat["models"] = time.perf_counter() - t
        out: dict = {"p0": p0}

        S = I = None
        if self.nn is not None:
            t = time.perf_counter()
            S, I = self.nn.search(emb, ticket_ids)
            out["p2"] = self.nn.vote(S, I)
            lat["neighbours"] = time.perf_counter() - t
        out["S"], out["I"] = S, I

        out["fallback"] = {k: apply_temperature(p0[k], self.calib["fallback"][k]["T"]) for k in TASKS}
        if self.mode == "full":
            t = time.perf_counter()
            X1 = self.m1.transform(texts)
            lat["tfidf"] = time.perf_counter() - t
            t = time.perf_counter()
            out["p1"] = self.m1.predict_proba(X1, self.labels)
            lat["models"] += time.perf_counter() - t
            t = time.perf_counter()
            out["ens_raw"] = self.ens.predict_proba([p0, out["p1"], out["p2"]], self.labels)
            out["final"] = {k: apply_temperature(out["ens_raw"][k], self.calib["full"][k]["T"]) for k in TASKS}
            lat["ensemble"] = time.perf_counter() - t
        else:
            out["final"] = out["fallback"]

        t = time.perf_counter()
        kf, hits = self.kw.features(texts)
        out["critical_score"] = self.crit.score(emb, kf)
        out["keyword_hits"] = hits
        lat["critical"] = time.perf_counter() - t
        lat["total"] = time.perf_counter() - t0
        out["latency_s"] = lat
        return out

    def predict_batch(self, subjects: list[str | None], bodies: list[str],
                      ticket_ids: list[str] | None = None) -> list[dict]:
        """Batch prediction. `ticket_ids` (optional) lets neighbour votes skip a ticket's own entry."""
        texts = build_texts(subjects, bodies)
        a = self.predict_arrays(texts, ticket_ids)
        n = len(texts)
        lat_ms = {k: round(v * 1000 / n, 2) for k, v in a["latency_s"].items()}
        thr = self.thresholds["critical"]["threshold"]
        conf_thr = self.thresholds[self.mode]
        nc_top = self.meta["neighbours"].get("top_n_display", 5)
        results = []
        for i in range(n):
            r: dict = {"mode": self.mode}
            for t in TASKS:
                p = a["final"][t][i]
                j = int(p.argmax())
                r[t] = {"label": self.labels[t][j], "confidence": round(float(p[j]), 4),
                        "probs": {l: float(v) for l, v in zip(self.labels[t], p)}}
            s = float(a["critical_score"][i])
            r["critical"] = {"score": round(s, 4), "threshold": round(thr, 4), "fires": bool(s >= thr),
                             "keyword_hits": a["keyword_hits"][i]}
            r["neighbours"] = (self.nn.describe(a["S"][i], a["I"][i], nc_top) if self.nn is not None
                               else {"top": [], "explanation": {}})
            r["suggested_conf_thresholds"] = dict(conf_thr)
            r["latency_ms"] = lat_ms
            results.append(r)
        return results

    def predict(self, subject: str | None, body: str) -> dict:
        """One live ticket. Empty/missing subject is fine; empty subject AND body raises ValueError."""
        return self.predict_batch([subject], [body])[0]
