"""Okapi BM25 keyword retrieval.

Dense-only retrieval fails exactly where document QA gets used: error codes,
version numbers, SKUs, acronyms, rare proper nouns. BM25 covers that failure
class with pure keyword scoring; the hybrid retriever fuses both ranked
lists. No dependencies beyond the standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Conservative suffix stripping: plurals and common verb endings only.

    Deliberately minimal - enough that "reduces" matches "reduce" and
    "metrics" matches "metric", conservative enough that "this"/"is"/"us"
    survive untouched.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """The one tokenizer shared by BM25, the reranker, and entailment."""
    return [stem(t) for t in _TOKEN_RE.findall(text.lower())]


class BM25Index:
    """In-memory BM25 index over the same records as the vector store."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._doc_freq: Counter = Counter()
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._term_freqs)

    def add(self, documents: list[str]) -> None:
        for doc in documents:
            tokens = tokenize(doc)
            self._term_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))
        self._rebuild_stats()

    def _rebuild_stats(self) -> None:
        self._doc_freq = Counter()
        for tf in self._term_freqs:
            self._doc_freq.update(tf.keys())
        self._avgdl = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

    def score(self, query: str, doc_index: int) -> float:
        tf = self._term_freqs[doc_index]
        dl = self._doc_lengths[doc_index]
        n_docs = len(self._term_freqs)
        total = 0.0
        for term in set(tokenize(query)):
            f = tf.get(term, 0)
            if f == 0:
                continue
            df = self._doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            total += idf * f * (self.k1 + 1) / denom
        return total

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs, best first, over all docs."""
        if not self._term_freqs:
            return []
        scored = [(i, self.score(query, i)) for i in range(len(self._term_freqs))]
        scored.sort(key=lambda item: -item[1])
        return scored[: max(1, min(k, len(scored)))]
