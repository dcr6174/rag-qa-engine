"""The end-to-end pipeline: ingest, hybrid retrieve, rerank, abstain, answer.

v0.3 turns the demo into an evaluation-first RAG system:

* structure-aware chunking with breadcrumbs and stable offsets
* hybrid BM25 + dense retrieval fused with RRF, then reranked
* threshold-based abstention - "this isn't in your documents" is a first-class answer
* sentence-level citations with entailment checks before display
* content-hash incremental indexing - re-indexing embeds only what changed
* numpy brute-force search, the right answer under roughly 100k chunks
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import split_markdown
from .citations import ABSTENTION_THRESHOLD, Citation, link_citations, should_abstain
from .embeddings import get_embedder
from .generate import default_generator, rewrite_query
from .rerank import get_reranker
from .retrieval import HybridRetriever, RetrievalResult
from .store import VectorStore

CHUNKER_VERSION = "markdown-v2"

import re

_SOURCE_MARK_RE = re.compile(r"\s*\[source:[^\]]*\]")


@dataclass
class AnswerResult:
    """Everything the answer screen needs to show its work."""

    text: str
    abstained: bool
    rewritten_question: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[RetrievalResult] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    unsupported_sentences: int = 0


class RAGPipeline:
    """Ingest documents, retrieve relevant passages, and answer questions."""

    def __init__(
        self,
        embedder_kind: str | None = None,
        generator: object | None = None,
        reranker_kind: str | None = None,
        chunk_size: int = 600,
        overlap: int = 100,
        top_k: int = 4,
        abstention_threshold: float = ABSTENTION_THRESHOLD,
    ) -> None:
        self.embedder = get_embedder(embedder_kind or os.environ.get("RAG_EMBEDDER", "hashing"))
        self.generator = generator or default_generator()
        self.reranker = get_reranker(reranker_kind or os.environ.get("RAG_RERANKER", "lexical"))
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.abstention_threshold = abstention_threshold
        self.store = VectorStore()
        self.retriever = HybridRetriever(self.store, self.reranker)
        self.documents: dict[str, str] = {}  # source -> original text (offset validation)

    # ---- ingestion -----------------------------------------------------

    def store_meta(self) -> dict:
        """Build settings persisted with the store and checked on load."""
        return {
            "embedder": type(self.embedder).__name__,
            "embedding_dim": self.embedder.dim,
            "tokenizer": "blake2b-bow" if type(self.embedder).__name__ == "HashingEmbedder" else "model",
            "chunker": f"{CHUNKER_VERSION}:{self.chunk_size}:{self.overlap}",
        }

    def ingest_text(self, text: str, source: str, page: int | None = None, replace: bool = True) -> int:
        """Chunk, embed, and index one document. Returns chunks actually embedded.

        Re-ingesting a source replaces its old chunks; unchanged chunks keep
        their stored vectors (content-hash incremental indexing), so only new
        or edited passages cost embeddings. ``replace=False`` appends, which
        is how multi-page documents (PDFs) add one page at a time.
        """
        chunks = split_markdown(text, source, self.chunk_size, self.overlap)
        if not chunks:
            return 0

        reusable = self.store.vectors_for_hashes({c.content_hash for c in chunks})
        if replace:
            self.store.remove_source(source)

        new_chunks = [c for c in chunks if c.content_hash not in reusable]
        embedded: dict[str, object] = {}
        if new_chunks:
            vectors = self.embedder.embed([c.embed_text for c in new_chunks])
            embedded = {c.content_hash: vectors[i] for i, c in enumerate(new_chunks)}

        import numpy as np

        all_vectors = np.vstack(
            [reusable.get(c.content_hash, embedded.get(c.content_hash)) for c in chunks]
        )
        records = []
        for c in chunks:
            records.append(
                {
                    "text": c.text,
                    "source": c.source,
                    "metadata": {
                        "chunk_index": c.index,
                        "start": c.start,
                        "end": c.end,
                        "breadcrumb": c.breadcrumb,
                        "kind": c.kind,
                        "parent_text": c.parent_text,
                        "content_hash": c.content_hash,
                        "embed_text": c.embed_text,
                        **({"page": page} if page is not None else {}),
                    },
                }
            )
        self.store.add(all_vectors, records)
        self.retriever.sync_bm25()
        if page is None:
            self.documents[source] = text
        else:
            self.documents[f"{source}#p{page}"] = text
        return len(new_chunks)

    def ingest_paths(self, paths: list[str | Path]) -> int:
        """Ingest every supported file in the given files or directories."""
        total = 0
        for path in paths:
            path = Path(path)
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if not file.is_file():
                    continue
                suffix = file.suffix.lower()
                if suffix in {".md", ".txt"}:
                    total += self.ingest_text(file.read_text(encoding="utf-8"), file.name)
                elif suffix == ".pdf":
                    total += self.ingest_pdf(file)
        return total

    def ingest_pdf(self, path: str | Path) -> int:
        """Ingest a PDF file with page-anchored chunks."""
        from .pdf import extract_pages

        path = Path(path)
        pages = extract_pages(path)
        return self._ingest_pages(pages, path.name)

    def ingest_pdf_pages(self, pages, source: str) -> int:
        """Ingest already-extracted PDF pages (e.g. from uploaded bytes)."""
        return self._ingest_pages(pages, source)

    def _ingest_pages(self, pages, source: str) -> int:
        self.store.remove_source(source)
        for key in [k for k in self.documents if k == source or k.startswith(f"{source}#p")]:
            del self.documents[key]
        total = 0
        for position, page in enumerate(pages):
            total += self.ingest_text(page.text, source, page=page.page, replace=position == 0)
        return total

    # ---- retrieval + answering ------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        query_vector = self.embedder.embed([question])[0]
        return self.retriever.retrieve(question, query_vector, k or self.top_k)

    def answer(self, question: str, k: int | None = None, history: list[dict] | None = None) -> AnswerResult:
        standalone = rewrite_query(question, history)
        results, all_candidates = self.retrieve(standalone, k)
        trace = [r.trace() for r in all_candidates[:10]]

        if should_abstain(results, self.abstention_threshold):
            return AnswerResult(
                text="",
                abstained=True,
                rewritten_question=standalone,
                sources=results,
                trace=trace,
            )

        text = self.generator.generate(standalone, results)
        # Generators mark provenance inline as [source: name]; the citation
        # layer carries that structurally, so strip the markers before the
        # answer (and entailment scores) are computed over clean sentences.
        text = _SOURCE_MARK_RE.sub("", text)
        citations = link_citations(text, results, self.documents or None)
        return AnswerResult(
            text=text,
            abstained=False,
            rewritten_question=standalone,
            citations=citations,
            sources=results,
            trace=trace,
            unsupported_sentences=sum(1 for c in citations if not c.entailed),
        )

    # ---- persistence -----------------------------------------------------

    def save_store(self, directory: str | Path) -> None:
        self.store.save(directory, meta=self.store_meta())

    def load_store(self, directory: str | Path) -> None:
        """Load a persisted store; refuses one built with different settings."""
        self.store = VectorStore.load(directory, expected_meta=self.store_meta())
        self.retriever = HybridRetriever(self.store, self.reranker)
        self.retriever.sync_bm25()
