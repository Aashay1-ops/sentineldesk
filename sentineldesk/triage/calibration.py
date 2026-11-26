"""Temperature scaling, NLL, ECE, reliability-diagram data, suggested confidence thresholds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-6


def safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """p_T = softmax(log p / T)."""
    return softmax(safe_log(probs) / T)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """T minimizing NLL on a held-out slice (half B). Bounds 0.05-20."""
    res = minimize_scalar(lambda T: nll(apply_temperature(probs, T), y), bounds=(0.05, 20.0), method="bounded")
    T = float(res.x)
    # Guard: never return a T that is worse than no scaling.
    return T if nll(apply_temperature(probs, T), y) <= nll(probs, y) else 1.0


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Per-bin confidence, accuracy and count (equal-width bins over [0, 1])."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = {"edges": edges.tolist(), "conf": [], "acc": [], "count": []}
    for b in range(n_bins):
        m = idx == b
        out["count"].append(int(m.sum()))
        out["conf"].append(float(conf[m].mean()) if m.any() else None)
        out["acc"].append(float(correct[m].mean()) if m.any() else None)
    return out


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    r, n = reliability(probs, y, n_bins), len(y)
    return float(sum(c / n * abs(a - k) for c, a, k in zip(r["count"], r["acc"], r["conf"]) if c))


def suggest_conf_threshold(probs: np.ndarray, y: np.ndarray, n_bins: int = 15, min_count: int = 10) -> float:
    """Confidence below which binned accuracy drops under 50%: the upper edge of the highest
    (well-populated) bin whose accuracy is < 0.5. If no bin is under 50%, the lowest observed confidence."""
    r = reliability(probs, y, n_bins)
    bad = [b for b in range(n_bins) if r["count"][b] >= min_count and r["acc"][b] < 0.5]
    if not bad:
        return float(np.round(probs.max(axis=1).min(), 4))
    return float(np.round(r["edges"][max(bad) + 1], 4))
