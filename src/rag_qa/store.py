"""In-memory vector store with cosine-similarity search and disk persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    """One retrieved passage."""

    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Keeps normalized embeddings in a matrix; search is a single dot product."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._records: list[dict] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        """Add one record (``text``, ``source``, optional ``metadata``) per vector."""
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have the same length")
        if self._vectors is None:
            self._vectors = vectors.astype(np.float32)
        else:
            if vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("embedding dimension mismatch")
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._records.extend(records)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        """Return the top-k records by cosine similarity (vectors are normalized)."""
        if self._vectors is None or len(self._records) == 0:
            return []
        k = max(1, min(k, len(self._records)))
        scores = self._vectors @ query_vector.astype(np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchResult(
                text=self._records[i]["text"],
                source=self._records[i]["source"],
                score=float(scores[i]),
                metadata=self._records[i].get("metadata", {}),
            )
            for i in top
        ]

    def save(self, directory: str | Path) -> None:
        """Persist vectors (``vectors.npy``) and records (``records.jsonl``)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise ValueError("cannot save an empty store")
        np.save(directory / "vectors.npy", self._vectors)
        with (directory / "records.jsonl").open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        store = cls()
        store._vectors = np.load(directory / "vectors.npy")
        with (directory / "records.jsonl").open(encoding="utf-8") as fh:
            store._records = [json.loads(line) for line in fh if line.strip()]
        return store
