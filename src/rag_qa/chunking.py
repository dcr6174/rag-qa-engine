"""Split documents into overlapping text chunks for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SEPARATOR_RE = re.compile(r"\n{2,}")


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    text: str
    source: str
    index: int
    metadata: dict = field(default_factory=dict)


def split_text(text: str, source: str, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    """Split *text* into chunks of roughly *chunk_size* characters.

    Paragraph boundaries are preferred. Consecutive chunks share *overlap*
    characters so sentences cut at a boundary stay recoverable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _SEPARATOR_RE.split(text) if p.strip()]

    chunks: list[Chunk] = []
    current = ""
    for para in paragraphs:
        # A paragraph longer than chunk_size is hard-split on its own.
        while len(para) > chunk_size:
            piece = para[:chunk_size]
            cut = piece.rfind(" ")
            if cut < chunk_size // 2:
                cut = chunk_size
            head, para = para[:cut].strip(), para[cut:].strip()
            current = _flush(chunks, current, head, source, overlap)
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            current = _flush(chunks, current, para, source, overlap)
    if current:
        chunks.append(Chunk(text=current, source=source, index=len(chunks)))
    # Re-index so chunk.index is stable per document regardless of flush path.
    return [Chunk(text=c.text, source=c.source, index=i, metadata=c.metadata) for i, c in enumerate(chunks)]


def _flush(chunks: list[Chunk], current: str, nxt: str, source: str, overlap: int) -> str:
    """Append *current* to chunks and seed the next buffer with tail overlap."""
    if current:
        chunks.append(Chunk(text=current, source=source, index=len(chunks)))
        tail = current[-overlap:] if overlap else ""
        return f"{tail}\n\n{nxt}".strip() if tail else nxt
    return nxt
