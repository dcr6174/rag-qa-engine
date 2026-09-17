"""PDF ingestion with page-anchored citations.

Page numbers are what makes a citation checkable in a real document, so every
chunk extracted from a PDF carries the 1-based page it came from and the UI
renders citations as ``file.pdf · p. 3``. Text extraction uses pypdf (pure
Python, no system dependencies). Scanned PDFs without a text layer yield no
chunks; the API reports that honestly instead of pretending to OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfPage:
    page: int  # 1-based
    text: str


def extract_pages(path: str | Path) -> list[PdfPage]:
    """Extract text per page from a PDF file."""
    from pypdf import PdfReader  # imported lazily; required only for PDFs

    reader = PdfReader(str(path))
    pages: list[PdfPage] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PdfPage(page=number, text=text))
    return pages


def extract_pages_from_bytes(data: bytes) -> list[PdfPage]:
    """Extract text per page from in-memory PDF bytes (browser uploads)."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[PdfPage] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PdfPage(page=number, text=text))
    return pages
