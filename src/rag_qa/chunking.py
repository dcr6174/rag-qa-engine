"""Structure-aware Markdown chunking with stable source offsets.

Documents are parsed into structural blocks (headings, paragraphs, tables,
fenced code, lists) that keep their exact character offsets into the original
file text. Chunks are then assembled per section:

* Split on the heading hierarchy. Every chunk carries its heading breadcrumb
  (``"Deployment > Configuration > Networking"``) and that breadcrumb is
  prefixed to the embedded text, so a chunk that reads "Timeout is 30s"
  retrieves as "Deployment > Configuration > Networking: Timeout is 30s".
* Tables and fenced code blocks are never split. An oversized table or code
  block is kept whole (and may exceed the target size) rather than broken.
* Chunks are small for retrieval precision; each chunk also carries the
  enclosing section text (``parent_text``) that the generator sees, so
  retrieval is precise while generation gets full context.
* ``start``/``end`` are absolute character offsets into the untouched
  original document text. Citation highlighting resolves through these
  offsets, so highlights never drift with cleaning or reformatting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

__all__ = ["Block", "Chunk", "parse_blocks", "split_markdown", "split_text"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


@dataclass
class Block:
    """One structural unit of a Markdown document."""

    kind: str  # heading | paragraph | table | code | list
    text: str
    start: int
    end: int
    level: int = 0  # heading level, 0 for non-headings


@dataclass
class Chunk:
    """A retrievable unit tied to exact offsets in the original document."""

    text: str
    source: str
    index: int
    start: int
    end: int
    breadcrumb: str = ""
    kind: str = "paragraph"
    parent_text: str = ""
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """The text actually embedded: breadcrumb-prefixed content."""
        return f"{self.breadcrumb}: {self.text}" if self.breadcrumb else self.text


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def parse_blocks(text: str) -> list[Block]:
    """Parse *text* into structural blocks with absolute character offsets."""
    blocks: list[Block] = []
    lines = text.split("\n")
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1  # account for the newline

    i = 0
    n = len(lines)

    def block_end(last_line: int) -> int:
        # end offset excludes the trailing newline of the last line
        return offsets[last_line] + len(lines[last_line])

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            blocks.append(Block("heading", heading.group(2), offsets[i], block_end(i), len(heading.group(1))))
            i += 1
            continue

        fence_open = _FENCE_RE.match(line)
        if fence_open:
            # Close on the same fence marker that opened the block: a ```
            # block must not be closed early by an unrelated ~~~ line (or
            # vice versa), which would otherwise split the block in half.
            close_re = re.compile(r"^\s*" + re.escape(fence_open.group(1)))
            j = i + 1
            while j < n and not close_re.match(lines[j]):
                j += 1
            j = min(j, n - 1)  # closing fence line, or EOF
            start, end = offsets[i], block_end(j)
            blocks.append(Block("code", text[start:end], start, end))
            i = j + 1
            continue

        if line.lstrip().startswith("|") and "|" in line.strip()[1:]:
            j = i
            while j + 1 < n and lines[j + 1].lstrip().startswith("|"):
                j += 1
            start, end = offsets[i], block_end(j)
            blocks.append(Block("table", text[start:end], start, end))
            i = j + 1
            continue

        if _LIST_RE.match(line):
            j = i
            while (
                j + 1 < n
                and lines[j + 1].strip()
                and (_LIST_RE.match(lines[j + 1]) or lines[j + 1].startswith((" ", "\t")))
            ):
                j += 1
            start, end = offsets[i], block_end(j)
            blocks.append(Block("list", text[start:end], start, end))
            i = j + 1
            continue

        # Paragraph: consecutive non-blank lines of no other kind.
        j = i
        while j + 1 < n:
            nxt = lines[j + 1]
            if not nxt.strip() or _HEADING_RE.match(nxt) or _FENCE_RE.match(nxt):
                break
            if nxt.lstrip().startswith("|") or _LIST_RE.match(nxt):
                break
            j += 1
        start, end = offsets[i], block_end(j)
        blocks.append(Block("paragraph", text[start:end], start, end))
        i = j + 1

    return blocks


@dataclass
class _Section:
    breadcrumb: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def _sections(blocks: list[Block]) -> list[_Section]:
    """Group content blocks under their heading-hierarchy breadcrumb."""
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current = _Section(breadcrumb="")

    for block in blocks:
        if block.kind == "heading":
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
            if current.blocks:
                sections.append(current)
            current = _Section(breadcrumb=" > ".join(title for _, title in heading_stack))
        else:
            current.blocks.append(block)
    if current.blocks:
        sections.append(current)
    return sections


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(block: Block, chunk_size: int) -> list[Block]:
    """Split an oversized paragraph at sentence boundaries, keeping offsets."""
    pieces: list[Block] = []
    start = 0
    text = block.text
    while len(text) - start > chunk_size:
        window = text[start : start + chunk_size]
        cuts = [m.end() for m in _SENTENCE_RE.finditer(window)]
        cut = cuts[-1] if cuts and cuts[-1] > chunk_size // 2 else None
        if cut is None:
            cut = window.rfind(" ")
            if cut < chunk_size // 2:
                cut = chunk_size
        piece_text = text[start : start + cut].strip()
        lead = len(text[start : start + cut]) - len(text[start : start + cut].lstrip())
        pieces.append(
            Block(
                "paragraph",
                piece_text,
                block.start + start + lead,
                block.start + start + lead + len(piece_text),
            )
        )
        start += cut
    tail = text[start:].strip()
    if tail:
        lead = len(text[start:]) - len(text[start:].lstrip())
        pieces.append(
            Block("paragraph", tail, block.start + start + lead, block.start + start + lead + len(tail))
        )
    return pieces


def split_markdown(
    text: str,
    source: str,
    chunk_size: int = 600,
    overlap: int = 100,
    parent_budget: int = 4000,
) -> list[Chunk]:
    """Split a Markdown document into structure-aware chunks.

    ``chunk_size`` and ``overlap`` are measured in characters of raw content
    (the breadcrumb prefix is added only at embedding time). Tables and code
    blocks are atomic: they are never split and never partially overlapped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    for section in _sections(parse_blocks(text)):
        # Pre-split oversized paragraphs; tables/code stay atomic.
        units: list[Block] = []
        for block in section.blocks:
            if block.kind in {"paragraph", "list"} and len(block.text) > chunk_size:
                units.extend(_split_long_paragraph(block, chunk_size))
            else:
                units.append(block)

        parent_text = section.text
        if len(parent_text) > parent_budget:
            parent_text = parent_text[:parent_budget].rsplit(" ", 1)[0] + " …"

        current: list[Block] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            chunk_text = "\n\n".join(b.text for b in current)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    index=len(chunks),
                    start=current[0].start,
                    end=current[-1].end,
                    breadcrumb=section.breadcrumb,
                    kind=current[0].kind if len(current) == 1 else "mixed",
                    parent_text=parent_text,
                    content_hash=_hash(chunk_text),
                )
            )
            # Character overlap seeded only from trailing paragraph text:
            # a tail that would cut into a table or code block is dropped.
            if overlap and current[-1].kind not in {"table", "code"}:
                # character overlap seeded only from trailing paragraph text;
                # any tail that would cut into a table or code block is dropped
                tail = chunk_text[-overlap:]
                # never seed a chunk mid-word: snap the tail to a word edge
                snap = tail.find(" ")
                if 0 <= snap < len(tail) - 1:
                    tail = tail[snap + 1 :]
                if tail.strip():
                    seed_start = current[-1].end - len(tail)
                    current = [Block("paragraph", tail, seed_start, current[-1].end)]
                    current_len = len(tail) + 2
                else:
                    current, current_len = [], 0
            else:
                current, current_len = [], 0

        for unit in units:
            add_len = len(unit.text) + (2 if current else 0)
            if current and current_len + add_len > chunk_size:
                flush()
                add_len = len(unit.text) + (2 if current else 0)
            current.append(unit)
            current_len += add_len
        current = [b for b in current if b.text.strip()]
        flush()

    return chunks


def split_text(text: str, source: str, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Chunk plain text or Markdown. Kept as the stable public entry point."""
    return split_markdown(text, source, chunk_size, overlap)
