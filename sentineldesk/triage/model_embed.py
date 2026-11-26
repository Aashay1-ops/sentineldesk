"""Model 0: one logistic regression per task on sentence embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sentineldesk.triage import TASKS
from sentineldesk.triage.labels import align_proba


class EmbedLR:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.clfs: dict[str, LogisticRegression] = {}

    def fit(self, X: np.ndarray, df: pd.DataFrame) -> "EmbedLR":
        for t in TASKS:
            self.clfs[t] = LogisticRegression(max_iter=2000, class_weight="balanced", C=self.C, random_state=42)
            self.clfs[t].fit(X, df[t].values)
        return self

    def predict_proba(self, X: np.ndarray, labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return {t: align_proba(c, c.predict_proba(X), labels[t]) for t, c in self.clfs.items()}
