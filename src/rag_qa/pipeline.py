"""Retriever, answer generator, and the end-to-end RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import split_text
from .embeddings import get_embedder
from .store import SearchResult, VectorStore


@dataclass
class Answer:
    """Pipeline output: the answer text plus the passages it rests on."""

    text: str
    sources: list[SearchResult]


class OfflineGenerator:
    """Compose an answer from retrieved passages. No external model needed.

    Picks the retrieved sentences with the most query-term overlap, so the
    output is always traceable to the sources it cites.
    """

    def generate(self, question: str, passages: list[SearchResult]) -> str:
        if not passages:
            return "I could not find anything relevant in the indexed documents."
        query_terms = {t for t in question.lower().split() if len(t) > 2}
        ranked: list[tuple[int, str, str]] = []
        for passage in passages:
            for sentence in _sentences(passage.text):
                overlap = len(query_terms & set(sentence.lower().split()))
                if overlap:
                    ranked.append((overlap, sentence, passage.source))
        if not ranked:
            best = passages[0]
            return f"{best.text.strip()} [source: {best.source}]"
        ranked.sort(key=lambda item: -item[0])
        parts, seen = [], set()
        for _, sentence, source in ranked:
            if sentence in seen:
                continue
            seen.add(sentence)
            parts.append(f"{sentence} [source: {source}]")
            if len(parts) == 3:
                break
        return " ".join(parts)


class OpenAICompatibleGenerator:
    """Generate with any OpenAI-compatible chat endpoint.

    Reads OPENAI_API_KEY, OPENAI_BASE_URL (defaults to https://api.openai.com/v1)
    and RAG_MODEL (defaults to gpt-4o-mini). Works with OpenAI, Azure OpenAI,
    Ollama, vLLM and similar servers. Keys come from the environment only and
    are never written to disk.
    """

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # optional dependency, imported lazily

        self._client = OpenAI()
        self._model = model or os.environ.get("RAG_MODEL", "gpt-4o-mini")

    def generate(self, question: str, passages: list[SearchResult]) -> str:
        context = "\n\n".join(
            f"[source: {p.source}]\n{p.text}" for p in passages
        )
        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. Cite the source "
                        "of every claim as [source: <name>]. If the context does "
                        "not contain the answer, say so."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return completion.choices[0].message.content.strip()


def _sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class RAGPipeline:
    """Ingest documents, retrieve relevant passages, and answer questions."""

    def __init__(
        self,
        embedder_kind: str | None = None,
        generator: object | None = None,
        chunk_size: int = 500,
        overlap: int = 80,
        top_k: int = 4,
    ) -> None:
        self.embedder = get_embedder(embedder_kind or os.environ.get("RAG_EMBEDDER", "hashing"))
        self.generator = generator or self._default_generator()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.store = VectorStore()

    @staticmethod
    def _default_generator():
        if os.environ.get("OPENAI_API_KEY"):
            try:
                return OpenAICompatibleGenerator()
            except ImportError:
                pass
        return OfflineGenerator()

    def ingest_text(self, text: str, source: str) -> int:
        """Chunk, embed, and index one document. Returns the chunk count."""
        chunks = split_text(text, source, self.chunk_size, self.overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        records = [
            {"text": c.text, "source": c.source, "metadata": {"chunk_index": c.index}}
            for c in chunks
        ]
        self.store.add(vectors, records)
        return len(chunks)

    def ingest_paths(self, paths: list[str | Path]) -> int:
        """Ingest every ``.md``/``.txt`` file in the given files or directories."""
        total = 0
        for path in paths:
            path = Path(path)
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if file.suffix.lower() in {".md", ".txt"} and file.is_file():
                    total += self.ingest_text(file.read_text(encoding="utf-8"), file.name)
        return total

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k or self.top_k)

    def answer(self, question: str, k: int | None = None) -> Answer:
        passages = self.retrieve(question, k)
        return Answer(text=self.generator.generate(question, passages), sources=passages)
