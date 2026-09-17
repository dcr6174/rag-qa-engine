from rag_qa.pipeline import RAGPipeline

DOC_A = (
    "Retrieval augmented generation combines a search index with a language model. "
    "The retriever fetches passages relevant to the question. "
    "The generator then writes an answer grounded in those passages.\n\n"
    "Chunking splits documents so each embedding covers one idea."
)
DOC_B = (
    "Banana bread needs ripe bananas, flour, sugar, and eggs. "
    "Bake at 175 degrees Celsius for about an hour."
)


def build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(embedder_kind="hashing")
    pipeline.ingest_text(DOC_A, "rag.md")
    pipeline.ingest_text(DOC_B, "bread.md")
    return pipeline


def test_ingest_counts_chunks():
    pipeline = build_pipeline()
    assert len(pipeline.store) >= 2


def test_retrieval_ranks_relevant_source_first():
    pipeline = build_pipeline()
    results = pipeline.retrieve("how does retrieval augmented generation work")
    assert results[0].source == "rag.md"
    assert results[0].score > results[-1].score


def test_answer_cites_sources_and_stays_on_topic():
    pipeline = build_pipeline()
    answer = pipeline.answer("what does the retriever do")
    assert "rag.md" in answer.text
    assert answer.sources, "answer should expose its supporting passages"
    assert "banana" not in answer.text.lower()


def test_answer_without_relevant_passages_is_honest():
    pipeline = RAGPipeline(embedder_kind="hashing")
    answer = pipeline.answer("anything at all")
    assert "could not find" in answer.text.lower()


def test_store_round_trip(tmp_path):
    pipeline = build_pipeline()
    pipeline.store.save(tmp_path)
    from rag_qa.store import VectorStore

    reloaded = VectorStore.load(tmp_path)
    assert len(reloaded) == len(pipeline.store)
