from conftest import make_pdf

from rag_qa.pipeline import RAGPipeline


def test_answer_includes_citations_sources_and_trace():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    result = pipeline.answer("How does RAG reduce hallucination?")
    assert not result.abstained
    assert result.citations
    assert result.sources
    assert result.trace
    assert {"bm25_rank", "dense_rank", "fused_rank", "final_rank", "band"} <= set(result.trace[0])


def test_unanswerable_question_abstains():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    result = pipeline.answer("Who won the 2019 cricket world cup?")
    assert result.abstained
    assert result.text == ""
    assert result.sources  # retrieved passages stay visible for transparency


def test_incremental_reindexing_embeds_only_what_changed(tmp_path):
    pipeline = RAGPipeline()
    first = pipeline.ingest_text("Timeout is 30 seconds by default.", "net.md")
    assert first > 0
    again = pipeline.ingest_text("Timeout is 30 seconds by default.", "net.md")
    assert again == 0  # identical content: nothing re-embedded
    changed = pipeline.ingest_text("Timeout is 45 seconds by default.", "net.md")
    assert changed > 0
    assert len(pipeline.store) == 1  # re-indexing replaces the source


def test_pdf_ingestion_anchors_pages():
    pipeline = RAGPipeline()
    from rag_qa.pdf import extract_pages_from_bytes

    pages = extract_pages_from_bytes(make_pdf(["Timeout is 30 seconds", "Retries use backoff"]))
    embedded = pipeline.ingest_pdf_pages(pages, "manual.pdf")
    assert embedded > 0
    result = pipeline.answer("What is the timeout?")
    assert not result.abstained
    assert any(c.page == 1 for c in result.citations if c.source == "manual.pdf")


def test_store_save_load_roundtrip(tmp_path):
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    pipeline.save_store(tmp_path / "store")
    fresh = RAGPipeline()
    fresh.load_store(tmp_path / "store")
    assert len(fresh.store) == len(pipeline.store)
    result = fresh.answer("What is mean reciprocal rank?")
    assert not result.abstained


def test_store_load_refuses_different_chunker(tmp_path):
    pipeline = RAGPipeline(chunk_size=600)
    pipeline.ingest_text("Some content to index for the store test.", "doc.md")
    pipeline.save_store(tmp_path / "store")
    other = RAGPipeline(chunk_size=900)
    import pytest

    with pytest.raises(Exception, match="different settings"):
        other.load_store(tmp_path / "store")


def test_history_rewrites_followups():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    result = pipeline.answer(
        "and the second stage?", history=[{"question": "What are the metrics of a RAG evaluation system?"}]
    )
    assert "system" in result.rewritten_question
