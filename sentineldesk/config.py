"""Central configuration. Every threshold the gate uses lives here so it can be explained to judges."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

HF_REPO = "Tobi-Bueck/customer-support-tickets"
HF_FILE = os.getenv("SENTINEL_DATA_FILE", "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
SEED = 42
KB_FRACTION = 0.8

# Embeddings: multilingual so a German ticket can retrieve English precedents (bonus: cross-lingual).
EMBED_MODEL = os.getenv("SENTINEL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = 5
PRECEDENT_VOTE_WEIGHT = float(os.getenv("SENTINEL_VOTE_WEIGHT", 0.5))  # tuned on KB validation slice

# LLM
LLM_MODEL = os.getenv("SENTINEL_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("SENTINEL_LLM_EFFORT", "low")  # short grounded drafts; keeps demo latency down

SUPPORTED_LANGS = {"en", "de"}

# ---- Escalation gate thresholds (tuned on a validation slice of the KB split, never the holdout) ----
GATE = {
    "min_type_conf": float(os.getenv("GATE_MIN_TYPE_CONF", 0.45)),
    "min_queue_conf": float(os.getenv("GATE_MIN_QUEUE_CONF", 0.30)),
    "min_top_similarity": float(os.getenv("GATE_MIN_SIM", 0.55)),
    # escalate if P(high) >= this AND Incident. Val slice: 0.30 -> recall 0.90 / precision 0.50 (0.50 -> recall 0.69)
    "high_priority_prob": float(os.getenv("GATE_HIGH_PROB", 0.30)),
    "min_body_chars": 15,
}

# ---- Outage / burst detection ----
BURST = {
    "window_seconds": 15 * 60,
    "similarity": 0.70,
    "min_cluster": 3,
}
