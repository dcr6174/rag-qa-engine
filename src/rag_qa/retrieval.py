"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion,
then reranked, with a full per-stage trace.

Dense-only retrieval fails on error codes, version numbers, SKUs, acronyms,
and rare proper nouns. Keyword-only retrieval misses paraphrases. Fusing both
ranked lists with RRF (about sixty lines, no tuning) fixes the most
embarrassing failure class of each. The fused top candidates are then
reranked; a reranker lifts quality more than any local embedding upgrade.

Every result carries its rank at each stage (keyword, dense, fused,
reranked) so the UI can show its work instead of one opaque score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25Index
from .rerank import band_for_score
from .store import VectorStore

RRF_K = 60  # standard RRF constant


@dataclass
class RetrievalResult:
    """One retrieved chunk with its position at every pipeline stage."""

    record_index: int
    text: str
    source: str
    metadata: dict
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int = 0
    band: str = "low"

    def trace(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.metadata.get("chunk_index"),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "fused_rank": self.fused_rank,
            "final_rank": self.final_rank,
            "band": self.band,
            "rerank_score": round(self.rerank_score, 4),
        }


class HybridRetriever:
    """BM25 + dense retrieval over one record set, fused and reranked."""

    def __init__(self, store: VectorStore, reranker=None) -> None:
        self.store = store
        self.bm25 = BM25Index()
        self.reranker = reranker

    def sync_bm25(self) -> None:
        """Rebuild the BM25 index from the store's current records."""
        self.bm25 = BM25Index()
        self.bm25.add([r.get("metadata", {}).get("embed_text", r["text"]) for r in self.store.records])

    def retrieve(
        self,
        query: str,
        query_vector,
        k: int = 4,
        candidates: int = 50,
        rerank: bool = True,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Return ``(top_k_results, all_scored_candidates)`` with stage traces."""
        n = len(self.store)
        if n == 0:
            return [], []
        candidate_cap = max(k, min(candidates, n))

        dense_hits = self.store.search(query_vector, k=candidate_cap)
        bm25_hits = self.bm25.search(query, k=candidate_cap)

        dense_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_hits)}
        bm25_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_hits)}

        # Reciprocal Rank Fusion over both lists.
        rrf: dict[int, float] = {}
        for idx, rank in dense_rank.items():
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for idx, rank in bm25_rank.items():
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        fused_order = sorted(rrf, key=lambda i: -rrf[i])
        results: list[RetrievalResult] = []
        for fused_pos, idx in enumerate(fused_order):
            record = self.store.records[idx]
            results.append(
                RetrievalResult(
                    record_index=idx,
                    text=record["text"],
                    source=record["source"],
                    metadata=record.get("metadata", {}),
                    bm25_rank=bm25_rank.get(idx),
                    dense_rank=dense_rank.get(idx),
                    fused_rank=fused_pos + 1,
                    rrf_score=rrf[idx],
                )
            )

        if rerank and self.reranker is not None and results:
            to_rerank = results[:candidate_cap]
            scores = self.reranker.score_pairs(
                query, [r.metadata.get("embed_text", r.text) for r in to_rerank]
            )
            for result, score in zip(to_rerank, scores):
                result.rerank_score = float(score)
                result.band = band_for_score(float(score))
            reranked = sorted(to_rerank, key=lambda r: -r.rerank_score)
            tail = results[candidate_cap:]
            results = reranked + tail
        else:
            for result in results:
                # No reranker: fused rank is the final order, band from RRF position.
                result.rerank_score = result.rrf_score * RRF_K  # ~1.0 at fused rank 1
                result.band = band_for_score(min(1.0, result.rerank_score))

        for final_pos, result in enumerate(results):
            result.final_rank = final_pos + 1
        return results[:k], results
