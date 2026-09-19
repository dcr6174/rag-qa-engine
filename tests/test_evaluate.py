from rag_qa.evaluate import evaluate, generate_qa_pairs, load_qa_pairs, sweep
from rag_qa.pipeline import RAGPipeline


def _pipeline():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    return pipeline


def test_metrics_on_fixture_corpus():
    report = evaluate(_pipeline(), load_qa_pairs("eval/qa_pairs.jsonl"))
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0
    assert 0.0 <= report.ndcg_at_k <= 1.0
    assert set(report.recall_at_k) == {1, 3, 5}
    assert report.faithfulness > 0.5


def test_false_answer_rate_uses_no_answer_subset():
    pairs = [
        {"question": "How does RAG reduce hallucination?", "expected_sources": ["rag_overview.md"]},
        {"question": "Who won the 2019 cricket world cup?", "expected_sources": [], "unanswerable": True},
    ]
    report = evaluate(_pipeline(), pairs)
    assert report.false_answer_rate == 0.0
    assert any(d.get("unanswerable") and d["abstained"] for d in report.details)


def test_ndcg_dedupes_chunks_from_one_source():
    # several chunks from the expected source in top-k must not inflate nDCG
    pairs = [{"question": "How does RAG reduce hallucination?", "expected_sources": ["rag_overview.md"]}]
    report = evaluate(_pipeline(), pairs)
    assert report.ndcg_at_k <= 1.0


def test_generated_qa_pairs_are_grounded_in_the_corpus():
    pipeline = _pipeline()
    pairs = generate_qa_pairs(pipeline, max_pairs=6)
    assert 1 <= len(pairs) <= 6
    sources = {r["source"] for r in pipeline.store.records}
    for pair in pairs:
        assert pair["expected_sources"][0] in sources


def test_sweep_returns_a_ranked_leaderboard():
    documents = _pipeline().documents
    pairs = load_qa_pairs("eval/qa_pairs.jsonl")
    leaderboard = sweep(dict(documents), pairs, max_configs=3)
    assert 1 <= len(leaderboard) <= 3
    ndcgs = [row["ndcg_at_k"] for row in leaderboard]
    assert ndcgs == sorted(ndcgs, reverse=True)
    for row in leaderboard:
        assert {"chunk_size", "overlap", "top_k", "hit_at_k", "mrr", "ndcg_at_k"} <= set(row)


def test_empty_qa_set_rejected():
    import pytest

    with pytest.raises(ValueError):
        evaluate(_pipeline(), [])
