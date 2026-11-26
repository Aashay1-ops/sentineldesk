"""Model 5: P(high-priority Incident) from embeddings + outage-keyword features.

The score feeds the escalation gate. It never overwrites predicted labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler


def critical_target(df: pd.DataFrame) -> np.ndarray:
    return ((df["type"] == "Incident") & (df["priority"] == "high")).to_numpy().astype(int)


class CriticalDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        self.threshold = 0.5

    def _X(self, emb: np.ndarray, kw_feats: np.ndarray) -> np.ndarray:
        return np.hstack([emb, self.scaler.transform(kw_feats)])

    def fit(self, emb: np.ndarray, kw_feats: np.ndarray, y: np.ndarray) -> "CriticalDetector":
        self.scaler.fit(kw_feats)
        self.clf.fit(self._X(emb, kw_feats), y)
        return self

    def score(self, emb: np.ndarray, kw_feats: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._X(emb, kw_feats))[:, 1]

    def tune_threshold(self, scores: np.ndarray, y: np.ndarray, target_recall: float) -> dict:
        """Highest threshold whose recall on this slice (half B) is still >= target_recall."""
        if y.sum() == 0:
            raise ValueError("No critical tickets in the tuning slice; cannot tune the threshold")
        prec, rec, thr = precision_recall_curve(y, scores)
        ok = np.where(rec[:-1] >= target_recall)[0]
        i = int(ok.max()) if len(ok) else 0
        self.threshold = float(thr[i])
        fires = scores >= self.threshold
        tp = int((fires & (y == 1)).sum())
        return {
            "threshold": self.threshold,
            "target_recall": target_recall,
            "half_b_recall": tp / int(y.sum()),
            "half_b_precision": tp / max(int(fires.sum()), 1),
            "half_b_positives": int(y.sum()),
        }
