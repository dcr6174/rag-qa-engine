"""Sentence-level citations and the offset property test.

The property that pays off: every citation offset must resolve to the text
actually quoted, in every generated citation, over the whole fixture corpus.
"""

import pytest

from rag_qa.citations import (
    entailment_score,
    link_citations,
    should_abstain,
    split_sentences,
)
from rag_qa.pipeline import RAGPipeline


def _results(pipeline, question, k=4):
    return pipeline.retrieve(question, k)[0]


def test_split_sentences_offsets_resolve():
    text = "RAG grounds answers. It retrieves passages first! Does it help?"
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == [
        "RAG grounds answers.",
        "It retrieves passages first!",
        "Does it help?",
    ]
    for s in sentences:
        assert text[s.start : s.end] == s.text


def test_entailment_scores_overlap():
    score = entailment_score(
        "RAG grounds answers in retrieved passages",
        "RAG grounds its answer in retrieved passages rather than memory",
    )
    assert score >= 0.5
    low = entailment_score("quantum entanglement enables teleportation", "the sky is blue")
    assert low < 0.5


def test_every_sentence_gets_a_citation():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    results = _results(pipeline, "How does RAG reduce hallucination?")
    answer = "RAG reduces hallucination through grounding. Answers come from retrieved text."
    citations = link_citations(answer, results)
    assert len(citations) == 2
    for citation in citations:
        assert citation.quote
        assert citation.doc_end > citation.doc_start


def test_citation_offsets_resolve_to_quoted_text_property():
    # property test: for every answer the pipeline produces on the fixture
    # corpus, every citation offset resolves to the exact quoted span in the
    # original document - highlights can never drift
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    questions = [
        "What are the three stages of a RAG system?",
        "How does RAG reduce hallucination?",
        "What does hit rate at k measure?",
        "What is mean reciprocal rank?",
        "Why use a hashing embedder in unit tests?",
    ]
    checked = 0
    for question in questions:
        result = pipeline.answer(question)
        assert not result.abstained
        for citation in result.citations:
            original = pipeline.documents[citation.source]
            assert original[citation.doc_start : citation.doc_end] == citation.quote
            checked += 1
    assert checked >= len(questions)


def test_offset_drift_is_caught_immediately():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    results = _results(pipeline, "How does RAG reduce hallucination?")
    corrupted = {source: "\u2588" * len(text) for source, text in pipeline.documents.items()}
    with pytest.raises(AssertionError, match="offset drift"):
        link_citations("RAG reduces hallucination through grounding.", results, corrupted)


def test_unsupported_sentences_are_flagged_not_hidden():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    results = _results(pipeline, "How does RAG reduce hallucination?")
    answer = "RAG reduces hallucination through grounding. The moon is made of green cheese."
    citations = link_citations(answer, results, pipeline.documents)
    assert citations[0].entailed
    assert not citations[1].entailed


def test_abstention_threshold_logic():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    answerable = _results(pipeline, "How does RAG reduce hallucination?")
    unanswerable = _results(pipeline, "Who won the 2019 cricket world cup?")
    assert not should_abstain(answerable)
    assert should_abstain(unanswerable)
    assert should_abstain([])
