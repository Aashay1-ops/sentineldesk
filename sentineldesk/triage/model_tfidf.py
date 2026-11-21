"""Model 1: word + char n-gram TF-IDF, one logistic regression per task."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

from sentineldesk.triage import TASKS
from sentineldesk.triage.labels import align_proba
from sentineldesk.triage.text import clean_for_features


def make_vectorizer(use_char: bool = True) -> FeatureUnion:
    parts = [("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                      max_features=200_000, lowercase=True))]
    if use_char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                              max_features=200_000, lowercase=True)))
    return FeatureUnion(parts)


class TfidfLR:
    def __init__(self, C: float = 4.0, use_char: bool = True, solver: str = "saga",
                 max_iter: int = 1000, tol: float = 1e-3):
        self.vec = make_vectorizer(use_char)
        self.C, self.solver, self.max_iter, self.tol = C, solver, max_iter, tol
        self.clfs: dict[str, LogisticRegression] = {}

    def fit(self, texts: list[str], df: pd.DataFrame) -> "TfidfLR":
        X = self.vec.fit_transform([clean_for_features(t) for t in texts])
        for t in TASKS:
            lr = LogisticRegression(C=self.C, class_weight="balanced", solver=self.solver,
                                    max_iter=self.max_iter, tol=self.tol, random_state=42)
            # liblinear is binary-only (multiclass support is removed in sklearn 1.8): wrap it as one-vs-rest.
            clf = OneVsRestClassifier(lr) if self.solver == "liblinear" else lr
            self.clfs[t] = clf.fit(X, df[t].values)
        return self

    def transform(self, texts: list[str]):
        return self.vec.transform([clean_for_features(t) for t in texts])

    def predict_proba(self, X, labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return {t: align_proba(c, c.predict_proba(X), labels[t]) for t, c in self.clfs.items()}
