"""Model 6: combine the probabilities of models 0, 1 and 2 per task (fitted on half A only).

method "pool" (default): weighted average of log-probabilities, p ∝ exp(Σ w_i log p_i), with
    3 weights per task (summing to 1) picked on half A by grid search for the best Macro-F1.
method "stack": a small logistic regression on the concatenated log-probabilities (the original plan).

Why "pool": half A has ~1.2k tickets; "stack" must learn 3 × n_classes inputs per class from them and
ended up BELOW TF-IDF alone. On half B (family split, i.e. new tickets) "pool" scored +5 points mean
Macro-F1 over "stack" (queue +7, priority +5), and won in every preprocessing variant tested.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sentineldesk.triage import TASKS
from sentineldesk.triage.calibration import safe_log, softmax
from sentineldesk.triage.labels import align_proba


def stack_features(prob_sets: list[dict[str, np.ndarray]], task: str) -> np.ndarray:
    return np.hstack([safe_log(p[task]) for p in prob_sets])


def weight_grid(n_models: int, step: float) -> list[tuple[float, ...]]:
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 6)
    return [w for w in itertools.product(ticks, repeat=n_models) if abs(sum(w) - 1) < 1e-6]


class Ensemble:
    def __init__(self, C: float = 1.0, method: str = "pool", grid_step: float = 0.1):
        if method not in ("pool", "stack"):
            raise ValueError(f"unknown ensemble method {method!r}")
        self.C, self.method, self.grid_step = C, method, grid_step
        self.clfs: dict[str, LogisticRegression] = {}
        self.weights: dict[str, tuple[float, ...]] = {}

    def fit(self, prob_sets: list[dict[str, np.ndarray]], y: dict[str, np.ndarray],
            labels: dict[str, list[str]]) -> "Ensemble":
        for t in TASKS:
            if self.method == "stack":
                clf = LogisticRegression(max_iter=2000, C=self.C, random_state=42)
                clf.fit(stack_features(prob_sets, t), np.asarray(labels[t])[y[t]])
                self.clfs[t] = clf
            else:
                logs = [safe_log(p[t]) for p in prob_sets]
                n_cls = len(labels[t])
                # grid order is deterministic, so ties always resolve the same way
                self.weights[t] = max(weight_grid(len(logs), self.grid_step), key=lambda w: f1_score(
                    y[t], sum(wi * l for wi, l in zip(w, logs)).argmax(1), average="macro",
                    labels=list(range(n_cls)), zero_division=0))
        return self

    def predict_proba(self, prob_sets: list[dict[str, np.ndarray]], labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
        if self.method == "stack":
            return {t: align_proba(c, c.predict_proba(stack_features(prob_sets, t)), labels[t]) for t, c in self.clfs.items()}
        return {t: softmax(sum(wi * safe_log(p[t]) for wi, p in zip(self.weights[t], prob_sets))) for t in TASKS}
