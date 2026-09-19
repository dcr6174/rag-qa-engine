"""Sentence-level citations, entailment checks, and abstention.

A citation card next to a paragraph does not prove the paragraph is
supported. So every answer *sentence* is linked to a concrete span in one
retrieved chunk - absolute character offsets into the original document -
and every sentence passes an entailment check against that span before
display. Sentences that fail are marked unsupported in the UI instead of
silently riding on the citation.

The offline entailment check is a transparent lexical proxy (content-word
overlap against the cited span), the same measure the eval harness reports
as faithfulness. It errs toward flagging, which is the honest direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .bm25 import tokenize
from .retrieval import RetrievalResult

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CONTENT_MIN_LEN = 4  # words shorter than this carry little claim

# A sentence is entailed when at least this fraction of its content words
# appear in the cited span. Matches the eval harness faithfulness threshold.
ENTAILMENT_THRESHOLD = 0.5

# Below this top reranker score the system abstains instead of answering.
ABSTENTION_THRESHOLD = 0.30


@dataclass
class Sentence:
    text: str
    start: int  # offset within the answer string
    end: int


@dataclass
class Citation:
    """One answer sentence linked to a span in an original document."""

    sentence: str
    sentence_start: int
    sentence_end: int
    source: str
    chunk_index: int
    quote: str  # exact text at doc_start:doc_end in the original document
    doc_start: int  # absolute offset into the original document
    doc_end: int
    entailment_score: float
    entailed: bool
    page: int | None = None
    final_rank: int = 0
    band: str = "low"

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "quote": self.quote,
            "doc_start": self.doc_start,
            "doc_end": self.doc_end,
            "entailment_score": round(self.entailment_score, 3),
            "entailed": self.entailed,
            "page": self.page,
            "final_rank": self.final_rank,
            "band": self.band,
        }


def split_sentences(text: str) -> list[Sentence]:
    """Split answer text into sentences with offsets into that text."""
    sentences: list[Sentence] = []
    pos = 0
    for piece in _SENTENCE_RE.split(text):
        idx = text.find(piece, pos)
        if idx == -1:
            continue
        stripped = piece.strip()
        lead = len(piece) - len(piece.lstrip())
        if stripped:
            sentences.append(Sentence(stripped, idx + lead, idx + lead + len(stripped)))
        pos = idx + len(piece)
    return sentences


def _content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) >= _CONTENT_MIN_LEN]


def entailment_score(sentence: str, span: str) -> float:
    """Fraction of the sentence's content words present in the cited span."""
    words = _content_words(sentence)
    if not words:
        return 1.0  # citation-only or numeric fragments carry no lexical claim
    span_terms = set(tokenize(span))
    return sum(1 for w in words if w in span_terms) / len(words)


def _best_span(sentence: str, chunk_text: str, window: int = 320) -> tuple[int, int, float]:
    """Find the chunk span with the highest entailment of *sentence*.

    Returns ``(start, end, score)`` as offsets within the chunk text.
    """
    sentence_terms = set(_content_words(sentence))
    if not sentence_terms:
        return 0, min(len(chunk_text), window), 1.0
    best = (0, min(len(chunk_text), window), -1.0)
    for sent in split_sentences(chunk_text):
        score = entailment_score(sentence, sent.text)
        if score > best[2]:
            best = (sent.start, sent.end, score)
    if best[2] >= 0:
        return best
    return 0, min(len(chunk_text), window), 0.0


def link_citations(
    answer: str,
    results: list[RetrievalResult],
    documents: dict[str, str] | None = None,
) -> list[Citation]:
    """Link each answer sentence to a span in the best-supporting chunk.

    ``documents`` maps source name to original document text; when provided,
    citation offsets are validated against the real document text so a drift
    bug surfaces immediately instead of months later.
    """
    citations: list[Citation] = []
    if not results:
        return citations
    for sent in split_sentences(answer):
        best: tuple[RetrievalResult, int, int, float] | None = None
        for result in results:
            start, end, score = _best_span(sent.text, result.text)
            if best is None or score > best[3]:
                best = (result, start, end, score)
        if best is None:
            continue
        result, span_start, span_end, score = best
        chunk_offset = result.metadata.get("start", 0)
        doc_start = chunk_offset + span_start
        doc_end = chunk_offset + span_end
        quote = result.text[span_start:span_end]
        if documents and result.source in documents:
            original = documents[result.source]
            if original[doc_start:doc_end] != quote:
                raise AssertionError(
                    f"citation offset drift in {result.source}: "
                    f"document[{doc_start}:{doc_end}] != quoted span"
                )
        citations.append(
            Citation(
                sentence=sent.text,
                sentence_start=sent.start,
                sentence_end=sent.end,
                source=result.source,
                chunk_index=result.metadata.get("chunk_index", 0),
                quote=quote,
                doc_start=doc_start,
                doc_end=doc_end,
                entailment_score=score,
                entailed=score >= ENTAILMENT_THRESHOLD,
                page=result.metadata.get("page"),
                final_rank=result.final_rank,
                band=result.band,
            )
        )
    return citations


def should_abstain(results: list[RetrievalResult], threshold: float = ABSTENTION_THRESHOLD) -> bool:
    """Abstain when the best reranked passage is below the relevance floor.

    The most valuable thing a QA tool can say is "this isn't in your
    documents". Gating on the top reranker score - a trained/coarse relevance
    judgment - is what separates a tool from a demo that always answers.
    """
    if not results:
        return True
    return results[0].rerank_score < threshold
