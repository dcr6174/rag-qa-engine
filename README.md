# rag-qa-engine

A small, honest retrieval-augmented generation (RAG) service in Python.
It ingests Markdown/text documents, retrieves the passages relevant to a
question, and answers with citations to the sources it used. An evaluation
harness measures retrieval quality and answer faithfulness on a labelled QA
set, and a pytest suite keeps the deterministic components under test.

The project runs **fully offline**: a deterministic hashing embedder and an
extractive answer generator need no downloads, API keys, or network. For
production-quality output, swap in neural embeddings
(`sentence-transformers`) and any OpenAI-compatible LLM endpoint with one
environment variable each.

## Architecture

```
 documents (.md/.txt)
        |
        v
  chunking.py        paragraph-aware splitting with overlap
        |
        v
  embeddings.py      HashingEmbedder (offline) or SentenceTransformerEmbedder
        |
        v
  store.py           normalized vectors in memory; cosine search; disk save/load
        |
        v            question
  pipeline.py  ----> retrieve top-k passages -> generate answer + citations
        |                      ^
        v                      |
  api.py (FastAPI)     evaluate.py (hit@k, MRR, faithfulness)
  /ingest /ask /health
```

Design choices:

- **Cited answers.** Every answer carries the passages behind it, so claims
  are auditable. The generation prompt (when an LLM is configured) instructs
  the model to answer only from the retrieved context.
- **Provider-agnostic LLM layer.** `OpenAICompatibleGenerator` talks to any
  OpenAI-compatible endpoint (OpenAI, Azure OpenAI, Ollama, vLLM) configured
  purely through environment variables. No keys are ever written to disk.
  Without a key, `OfflineGenerator` composes an extractive answer from the
  highest-overlap retrieved sentences.
- **Offline-first testing.** The deterministic hashing embedder needs no
  model downloads, so unit and integration tests run in milliseconds in CI.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the test suite (offline, no keys needed)
PYTHONPATH=src pytest tests/ -q

# run the evaluation harness against the bundled corpus and QA set
python scripts/run_eval.py

# serve the API
uvicorn rag_qa.api:app --app-dir src --reload
```

Then:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How does RAG reduce hallucination?"}'
```

### Optional upgrades

```bash
pip install sentence-transformers   # neural embeddings
pip install openai                  # LLM generation

export RAG_EMBEDDER=sentence-transformers
export OPENAI_API_KEY=...           # or point OPENAI_BASE_URL at a local server
export RAG_MODEL=gpt-4o-mini
```

## Evaluation

`scripts/run_eval.py` scores the pipeline on the labelled set in
`eval/qa_pairs.jsonl` (questions with their expected source documents) and
reports:

- **hit@k** - how often an expected source appears in the top-k passages
- **MRR** - mean reciprocal rank of the first expected source
- **faithfulness** - fraction of answer sentences supported by the retrieved
  passages (a transparent lexical proxy, not a neural judge)

Current offline baseline (hashing embedder, extractive generator, 9 chunks
from `sample_data/`, 8 questions): **hit@4 = 1.00, MRR = 0.81,
faithfulness = 0.97**. Reproduce with `python scripts/run_eval.py`.

## Project layout

```
src/rag_qa/
  chunking.py     paragraph-aware splitter with character overlap
  embeddings.py   hashing + sentence-transformers embedders (L2-normalized)
  store.py        in-memory vector store, cosine search, save/load
  pipeline.py     retriever, offline + OpenAI-compatible generators, RAGPipeline
  api.py          FastAPI service: /health, /ingest, /ask
  evaluate.py     hit@k, MRR, faithfulness over a labelled QA set
scripts/run_eval.py
tests/            14 tests: chunking, embeddings, pipeline, evaluation
sample_data/      three documents the demo indexes
eval/qa_pairs.jsonl
```

## Testing strategy

- **Unit tests** pin deterministic behavior: chunk size limits and overlap,
  normalized and reproducible embeddings, similarity ordering.
- **Integration tests** run the whole pipeline on a fixed corpus and assert
  the right document ranks first and answers cite their sources.
- **Regression evaluation** re-scores the labelled QA set after any change
  to chunking, embeddings, or generation; metric drops are treated like
  failing tests.

## Roadmap

- FAISS or Chroma backend for larger corpora
- Hybrid lexical + vector retrieval (BM25 + embeddings)
- Cross-encoder re-ranking of retrieved passages
- LLM-as-judge faithfulness scoring alongside the lexical proxy
- Docker image and CI workflow

## License

MIT - see [LICENSE](LICENSE).
