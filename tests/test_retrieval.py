import numpy as np

from rag_qa.embeddings import HashingEmbedder
from rag_qa.rerank import LexicalReranker
from rag_qa.retrieval import HybridRetriever
from rag_qa.store import VectorStore


def _build(texts):
    store = VectorStore()
    embedder = HashingEmbedder()
    vectors = embedder.embed(texts)
    records = [
        {"text": t, "source": f"doc{i}.md", "metadata": {"chunk_index": i, "embed_text": t}}
        for i, t in enumerate(texts)
    ]
    store.add(vectors, records)
    retriever = HybridRetriever(store, LexicalReranker())
    retriever.sync_bm25()
    return retriever, embedder


def test_hybrid_finds_error_code_dense_retrieval_misses():
    texts = [
        "error code E42 signals an upstream timeout in the gateway",
        "general guidance about handling failures and trying again later",
        "a paragraph about timeouts and reliability in distributed systems",
    ]
    retriever, embedder = _build(texts)
    qv = embedder.embed(["what does E42 mean"])[0]
    top, _ = retriever.retrieve("what does E42 mean", qv, k=1)
    assert top[0].text == texts[0]


def test_every_result_carries_a_full_stage_trace():
    retriever, embedder = _build(["alpha beta gamma", "delta epsilon zeta"])
    qv = embedder.embed(["alpha"])[0]
    top, all_candidates = retriever.retrieve("alpha", qv, k=1)
    trace = top[0].trace()
    for field in ("bm25_rank", "dense_rank", "fused_rank", "final_rank", "band", "rerank_score"):
        assert field in trace
    assert top[0].final_rank == 1
    assert len(all_candidates) == 2


def test_rrf_rewards_appearing_in_both_lists():
    texts = ["shared keywords here", "shared keywords there", "other words entirely"]
    retriever, embedder = _build(texts)
    qv = embedder.embed(["shared keywords"])[0]
    _, all_candidates = retriever.retrieve("shared keywords", qv, k=2, rerank=False)
    rrf_scores = [r.rrf_score for r in all_candidates]
    assert rrf_scores == sorted(rrf_scores, reverse=True)


def test_empty_store_returns_nothing():
    retriever = HybridRetriever(VectorStore(), LexicalReranker())
    top, all_candidates = retriever.retrieve("q", np.zeros(384, dtype=np.float32))
    assert top == [] and all_candidates == []
