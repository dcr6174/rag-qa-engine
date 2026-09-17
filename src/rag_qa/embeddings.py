"""Embedding backends.

Two interchangeable embedders:

* ``HashingEmbedder`` - deterministic bag-of-words hashing. Zero downloads,
  zero network: the whole project runs and tests offline with it.
* ``SentenceTransformerEmbedder`` - neural embeddings via the optional
  ``sentence-transformers`` package (default model: all-MiniLM-L6-v2).

Both return L2-normalized vectors, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic lexical embedder (no external dependencies)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, slot] += sign
        return _normalize(vectors)


class SentenceTransformerEmbedder:
    """Neural embedder backed by sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts), dtype=np.float32))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def get_embedder(kind: str = "hashing", **kwargs):
    """Factory: ``kind`` is ``hashing`` or ``sentence-transformers``."""
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    if kind in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder kind: {kind!r}")
