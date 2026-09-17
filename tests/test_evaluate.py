from rag_qa.evaluate import evaluate
from rag_qa.pipeline import RAGPipeline


def test_evaluate_reports_perfect_score_on_matching_corpus():
    pipeline = RAGPipeline(embedder_kind="hashing")
    pipeline.ingest_text(
        "RAG stands for retrieval augmented generation. It grounds answers in documents.",
        "rag.md",
    )
    pairs = [{"question": "what does RAG stand for", "expected_sources": ["rag.md"]}]
    report = evaluate(pipeline, pairs, k=1)
    assert report.questions == 1
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0
    assert 0.0 <= report.faithfulness <= 1.0


def test_evaluate_reports_miss_when_source_absent():
    pipeline = RAGPipeline(embedder_kind="hashing")
    pipeline.ingest_text("Completely unrelated content about gardening.", "garden.md")
    pairs = [{"question": "quantum computing", "expected_sources": ["quantum.md"]}]
    report = evaluate(pipeline, pairs, k=1)
    assert report.hit_at_k == 0.0
    assert report.mrr == 0.0
