"""In-memory vector store with disk persistence and store hygiene.

Search is a numpy brute-force dot product. Under roughly 100k chunks that is
the right answer: exact, instant at this scale, zero extra dependencies. If a
corpus outgrows it, sqlite-vec plus FTS5 gives vectors, keyword search, and
persistence in one file with no server - that trade-off is documented in the
README.

Store hygiene (unglamorous, necessary):

* ``store_meta.json`` persists the embedding model name, dimension,
  tokenizer, and chunker config alongside the vectors. Loading a store built
  with different settings raises :class:`StoreMismatchError` - silent model
  mismatch produces plausible garbage and is miserable to diagnose.
* Every record carries a content hash, so re-indexing embeds only what
  changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STORE_FORMAT_VERSION = 2


class StoreMismatchError(RuntimeError):
    """Raised when a persisted store's build settings differ from the loader's."""

    def __init__(self, mismatches: dict[str, dict[str, object]]) -> None:
        self.mismatches = mismatches
        fields = ", ".join(
            f"{name} (store={pair['store']!r}, current={pair['current']!r})"
            for name, pair in mismatches.items()
        )
        super().__init__(
            f"store was built with different settings: {fields}. "
            "Re-index the documents instead of loading this store."
        )


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

    @property
    def records(self) -> list[dict]:
        return self._records

    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        """Add one record (``text``, ``source``, ``metadata``) per vector."""
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have the same length")
        if self._vectors is None:
            self._vectors = vectors.astype(np.float32)
        else:
            if vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("embedding dimension mismatch")
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._records.extend(records)

    def known_hashes(self) -> set[str]:
        """Content hashes already indexed (for incremental re-indexing)."""
        return {r.get("metadata", {}).get("content_hash", "") for r in self._records}

    def vectors_for_hashes(self, hashes: set[str]) -> dict[str, np.ndarray]:
        """Map content hash to its stored vector (reuse on re-indexing)."""
        found: dict[str, np.ndarray] = {}
        if self._vectors is None:
            return found
        for i, record in enumerate(self._records):
            h = record.get("metadata", {}).get("content_hash")
            if h in hashes:
                found[h] = self._vectors[i]
        return found

    def remove_source(self, source: str) -> int:
        """Drop every record from *source* (used when re-indexing a file)."""
        keep = [i for i, r in enumerate(self._records) if r.get("source") != source]
        removed = len(self._records) - len(keep)
        if removed == 0:
            return 0
        self._records = [self._records[i] for i in keep]
        if self._vectors is not None:
            self._vectors = self._vectors[keep] if keep else None
        return removed

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[int, float]]:
        """Return ``(record_index, cosine_score)`` pairs, best first."""
        if self._vectors is None or len(self._records) == 0:
            return []
        k = max(1, min(k, len(self._records)))
        scores = self._vectors @ query_vector.astype(np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: str | Path, meta: dict | None = None) -> None:
        """Persist vectors, records, and build metadata for hygiene checks."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise ValueError("cannot save an empty store")
        np.save(directory / "vectors.npy", self._vectors)
        with (directory / "records.jsonl").open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        store_meta = {"format_version": STORE_FORMAT_VERSION, **(meta or {})}
        with (directory / "store_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(store_meta, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory: str | Path, expected_meta: dict | None = None) -> "VectorStore":
        """Load a store, refusing one built with different model/chunker settings."""
        directory = Path(directory)
        meta_path = directory / "store_meta.json"
        if expected_meta is not None:
            if not meta_path.exists():
                raise StoreMismatchError({"store_meta.json": {"store": "missing", "current": expected_meta}})
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            mismatches = {
                key: {"store": stored.get(key), "current": value}
                for key, value in expected_meta.items()
                if stored.get(key) != value
            }
            if mismatches:
                raise StoreMismatchError(mismatches)
        store = cls()
        store._vectors = np.load(directory / "vectors.npy")
        with (directory / "records.jsonl").open(encoding="utf-8") as fh:
            store._records = [json.loads(line) for line in fh if line.strip()]
        return store
