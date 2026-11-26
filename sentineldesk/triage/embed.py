"""Sentence-embedding wrapper shared by model 0, neighbour votes, the critical detector and retrieval."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Normalized embeddings; the same model + prefix must be used at train and predict time."""

    def __init__(self, model_name: str, prefix: str = "", batch_size: int = 64,
                 max_seq_length: int = 512, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name, self.prefix, self.batch_size = model_name, prefix, batch_size
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        if self.device == "cuda":
            self.model.half()
        log.info("embedder %s on %s (prefix=%r)", model_name, self.device, prefix)

    @property
    def dim(self) -> int:
        fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(fn())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            [self.prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return emb.astype(np.float32)
