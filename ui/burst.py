"""Live outage (burst) detection - README bonus feature.

Keeps a sliding window of recently processed tickets (query embedding + triage signals). When at least
`min_cluster` failure-like tickets inside the window are similar to the newest one (cosine >= `sim_threshold`),
the console raises a Major Incident banner. The threshold is placed by scripts/build_ui_report.py just above
the similarity of *distinct* hand-written issues, so unrelated tickets arriving together don't form a cluster.
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_STOP = set(
    "the and for with our are not has have this that from been was were but all any can our your you we ich wir die der das "
    "und ist sind nicht mit von für auf ein eine einen den dem des unser unsere seit dear team support customer kunden "
    "hello hallo please bitte issue problem since seit heute today morning".split()
)


@dataclass
class _Event:
    ts: float
    ref: str
    vec: np.ndarray
    subject: str
    text: str
    failure_like: bool


@dataclass
class BurstAlert:
    size: int
    ticket_refs: list[str]
    keywords: list[str]
    first_seen: float
    last_seen: float
    sample_subjects: list[str] = field(default_factory=list)


class BurstDetector:
    def __init__(self, window_minutes: float = 30.0, sim_threshold: float = 0.92, min_cluster: int = 3):
        self.window = window_minutes * 60
        self.sim_threshold = sim_threshold
        self.min_cluster = min_cluster
        self.events: list[_Event] = []
        self.active: BurstAlert | None = None
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        self.events = [e for e in self.events if now - e.ts <= self.window]
        if self.active and now - self.active.last_seen > self.window:
            self.active = None

    def add(self, ref: str, vec: np.ndarray, subject: str, text: str, failure_like: bool,
            now: float | None = None) -> BurstAlert | None:
        now = now or time.time()
        with self._lock:
            self._prune(now)
            self.events.append(_Event(now, ref, vec, subject, text, failure_like))
            if not failure_like:
                return None
            members = [e for e in self.events if e.failure_like and float(e.vec @ vec) >= self.sim_threshold]
            if len(members) < self.min_cluster:
                return None
            words = Counter(
                w for e in members for w in set(re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", (e.subject + " " + e.text).lower()))
                if w not in _STOP
            )
            self.active = BurstAlert(
                size=len(members),
                ticket_refs=[e.ref for e in members],
                keywords=[w for w, c in words.most_common(8) if c >= max(2, len(members) // 2)][:5],
                first_seen=min(e.ts for e in members),
                last_seen=now,
                sample_subjects=[e.subject for e in members if e.subject][:3],
            )
            return self.active

    def reset(self) -> None:
        with self._lock:
            self.events.clear()
            self.active = None
