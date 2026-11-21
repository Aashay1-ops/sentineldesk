"""Train every triage piece in the agreed order and save artifacts.

train  -> model 0, model 1, critical detector, neighbour index
half A -> ensemble (6)   (train / half A / half B are cut by paraphrase family: no reworded copies across them)
half B -> temperature scaling, critical threshold, suggested confidence thresholds
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sentineldesk.triage import TASKS
from sentineldesk.triage.calibration import apply_temperature, ece, fit_temperature, nll, suggest_conf_threshold
from sentineldesk.triage.config import REPO_ROOT
from sentineldesk.triage.critical import CriticalDetector, critical_target
from sentineldesk.triage.data import assert_disjoint
from sentineldesk.triage.ensemble import Ensemble
from sentineldesk.triage.keywords import KeywordMatcher
from sentineldesk.triage.labels import encode, make_labels, save_labels
from sentineldesk.triage.model_embed import EmbedLR
from sentineldesk.triage.model_tfidf import TfidfLR
from sentineldesk.triage.neighbours import NeighbourIndex
from sentineldesk.triage.text import build_texts

log = logging.getLogger(__name__)


class _Timer:
    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                log.info("▶ %s ...", name)

            def __exit__(self, *exc):
                timer.times[name] = round(time.perf_counter() - self.t0, 2)
                log.info("✔ %s: %.1fs", name, timer.times[name])

        return _Ctx()


def _calib_report(probs: np.ndarray, y: np.ndarray, T: float, bins: int, min_bin: int) -> dict:
    cal = apply_temperature(probs, T)
    return {
        "T": T,
        "nll_before": nll(probs, y), "nll_after": nll(cal, y),
        "ece_before": ece(probs, y, bins), "ece_after": ece(cal, y, bins),
        "accuracy": float((cal.argmax(1) == y).mean()),
        "macro_f1": float(f1_score(y, cal.argmax(1), average="macro", labels=list(range(probs.shape[1])), zero_division=0)),
        "suggested_conf_threshold": suggest_conf_threshold(cal, y, bins, min_bin),
    }


def train_all(cfg: dict, parts: dict[str, pd.DataFrame], embs: dict[str, np.ndarray], out_dir: Path,
              embed_meta: dict, test_ids: list[str] | None = None, fallback_only: bool = False,
              keywords_path: Path | None = None) -> dict:
    """parts/embs: 'train', 'half_a', 'half_b' (rows of embs aligned with parts). Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = _Timer()
    train, half_a, half_b = parts["train"], parts["half_a"], parts["half_b"]
    ids = {k: parts[k]["ticket_id"].tolist() for k in ("train", "half_a", "half_b")}
    if test_ids is not None:
        ids["test"] = list(test_ids)
    assert_disjoint(ids)

    labels = make_labels(train)
    for name in ("half_a", "half_b"):
        unseen = {t: set(parts[name][t]) - set(labels[t]) for t in TASKS}
        if any(unseen.values()):
            raise ValueError(f"{name} has labels not seen in train: {unseen}")
    save_labels(labels, out_dir / "labels.json")
    y = {k: {t: encode(parts[k][t], labels[t]) for t in TASKS} for k in ("train", "half_a", "half_b")}
    texts = {k: build_texts(parts[k]["subject"], parts[k]["body"]) for k in ("train", "half_a", "half_b")}
    bins, min_bin = cfg["calibration"]["ece_bins"], cfg["calibration"]["conf_threshold_min_bin"]
    calib: dict = {"fallback": {}, "full": {}}
    thresholds: dict = {"fallback": {}, "full": {}}

    # Model 0 + fallback calibration on half B
    with timer("model 0 (embeddings + LR)"):
        m0 = EmbedLR(C=cfg["model_embed"]["C"]).fit(embs["train"], train)
        joblib.dump(m0, out_dir / "model_embed.joblib")
        p0 = {k: m0.predict_proba(embs[k], labels) for k in ("half_a", "half_b")}
    for t in TASKS:
        T = fit_temperature(p0["half_b"][t], y["half_b"][t])
        calib["fallback"][t] = _calib_report(p0["half_b"][t], y["half_b"][t], T, bins, min_bin)
        thresholds["fallback"][t] = calib["fallback"][t]["suggested_conf_threshold"]

    # Critical detector (5)
    with timer("critical detector (5)"):
        kw = KeywordMatcher.from_yaml(keywords_path or (REPO_ROOT / cfg["keywords"]["path"]))
        joblib.dump(kw, out_dir / "keywords.joblib")
        kf_train, _ = kw.features(texts["train"])
        kf_b, _ = kw.features(texts["half_b"])
        crit = CriticalDetector().fit(embs["train"], kf_train, critical_target(train))
        thresholds["critical"] = crit.tune_threshold(crit.score(embs["half_b"], kf_b), critical_target(half_b),
                                                     cfg["critical"]["target_recall"])
        joblib.dump(crit, out_dir / "critical.joblib")
    log.info("critical: %s", thresholds["critical"])

    if not fallback_only:
        with timer("model 1 (TF-IDF + LR)"):
            mc = cfg["model_tfidf"]
            m1 = TfidfLR(C=mc["C"], use_char=mc["use_char"], solver=mc["solver"], max_iter=mc["max_iter"],
                         tol=mc["tol"]).fit(texts["train"], train)
            joblib.dump(m1, out_dir / "model_tfidf.joblib")
            p1 = {k: m1.predict_proba(m1.transform(texts[k]), labels) for k in ("half_a", "half_b")}

        with timer("model 2 (neighbour index + votes)"):
            nc = cfg["neighbours"]
            codes = {t: y["train"][t] for t in TASKS}
            nn = NeighbourIndex(embs["train"], ids["train"], codes, labels, nc["k"], nc["p"], nc["alpha"])
            np.savez(out_dir / "neighbours.npz", emb=embs["train"], ids=np.array(ids["train"]),
                     **{f"codes_{t}": codes[t] for t in TASKS})
            p2 = {}
            for k in ("half_a", "half_b"):
                S, I = nn.search(embs[k], ids[k])
                for r, qid in enumerate(ids[k]):  # a ticket must never vote for itself
                    assert qid not in {ids["train"][i] for i in I[r]}, f"{qid} is in its own neighbour list"
                p2[k] = nn.vote(S, I)

        with timer("ensemble (6) on half A + temperature on half B"):
            ec = cfg["ensemble"]
            ens = Ensemble(C=ec.get("C", 1.0), method=ec.get("method", "pool"), grid_step=ec.get("grid_step", 0.1)
                           ).fit([p0["half_a"], p1["half_a"], p2["half_a"]], y["half_a"], labels)
            joblib.dump(ens, out_dir / "ensemble.joblib")
            pe = ens.predict_proba([p0["half_b"], p1["half_b"], p2["half_b"]], labels)
            for t in TASKS:
                T = fit_temperature(pe[t], y["half_b"][t])
                calib["full"][t] = _calib_report(pe[t], y["half_b"][t], T, bins, min_bin)
                thresholds["full"][t] = calib["full"][t]["suggested_conf_threshold"]
                calib["full"][t]["ensemble_method"] = ens.method
                if ens.method == "pool":
                    calib["full"][t]["pool_weights_m0_m1_m2"] = list(ens.weights[t])

    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    meta = {
        **embed_meta,
        "neighbours": cfg["neighbours"],
        "sizes": {k: len(v) for k, v in ids.items()},
        "fallback_only": fallback_only,
        "train_times_s": timer.times,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"calibration": calib, "thresholds": thresholds, "times": timer.times}
