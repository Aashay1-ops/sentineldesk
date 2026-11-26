"""Live outage / burst detection: N semantically similar tickets in a short window -> Major Incident."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from . import config


class BurstDetector:
    def __init__(self, window=config.BURST["window_seconds"], sim=config.BURST["similarity"],
                 min_cluster=config.BURST["min_cluster"]):
        self.window, self.sim, self.min_cluster = window, sim, min_cluster
        self.events: deque = deque()
        self.lock = threading.Lock()

    def observe(self, vec: np.ndarray, label: str, now: float | None = None) -> dict | None:
        now = now or time.time()
        with self.lock:
            while self.events and now - self.events[0][0] > self.window:
                self.events.popleft()
            similar = [(t, lbl) for t, v, lbl in self.events if float(v @ vec) >= self.sim]
            self.events.append((now, vec, label))
        if len(similar) + 1 >= self.min_cluster:
            return {
                "count": len(similar) + 1,
                "window_min": self.window // 60,
                "examples": [lbl for _, lbl in similar[-4:]] + [label],
            }
        return None

    def reset(self):
        with self.lock:
            self.events.clear()
