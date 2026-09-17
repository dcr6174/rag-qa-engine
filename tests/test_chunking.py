"""Golden chunk boundaries, atomic blocks, breadcrumbs, and offset stability."""

import pytest

from rag_qa.chunking import parse_blocks, split_text
from conftest import FIXTURE_DOC


def test_golden_block_boundaries():
    kinds = [b.kind for b in parse_blocks(FIXTURE_DOC)]
    assert kinds == [
        "heading", "paragraph",
        "heading", "paragraph", "table", "paragraph",
        "heading", "paragraph", "code", "paragraph",
    ]


def test_golden_chunk_boundaries():
    chunks = split_text(FIXTURE_DOC, "guide.md", chunk_size=600, overlap=0)
    # every chunk offset resolves to exactly its text in the original document
    for chunk in chunks:
        assert FIXTURE_DOC[chunk.start : chunk.end] == chunk.text
    # the heading itself is never chunked as content
    assert all(not c.text.startswith("# ") for c in chunks)


def test_breadcrumbs_follow_heading_hierarchy():
    chunks = split_text(FIXTURE_DOC, "guide.md", chunk_size=600, overlap=0)
    crumbs = {c.breadcrumb for c in chunks}
    assert any("Deployment Guide > Configuration" in c for c in crumbs)
    assert any("Deployment Guide > Networking" in c for c in crumbs)
    net = next(c for c in chunks if "Networking" in c.breadcrumb)
    assert net.embed_text.startswith("Deployment Guide > Networking:")


def test_tables_are_never_split():
    chunks = split_text(FIXTURE_DOC, "guide.md", chunk_size=80, overlap=0)
    table_chunks = [c for c in chunks if "| Setting |" in c.text or "| workers |" in c.text]
    assert len(table_chunks) == 1  # atomic even over the size target
    assert "| debug   | false   |" in table_chunks[0].text


def test_code_blocks_are_never_split():
    chunks = split_text(FIXTURE_DOC, "guide.md", chunk_size=80, overlap=0)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1
    assert "client.retry(backoff=2)" in code_chunks[0].text


def test_long_paragraph_splits_at_sentence_boundaries():
    text = "First sentence here. Second sentence follows. Third sentence ends it. " * 20
    chunks = split_text(text, "long.txt", chunk_size=140, overlap=0)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.endswith(".")


def test_overlap_never_seeds_mid_word():
    text = "alpha beta gamma delta epsilon zeta eta theta. " * 30
    chunks = split_text(text, "overlap.txt", chunk_size=120, overlap=40)
    for chunk in chunks:
        first = chunk.text.split(" ", 1)[0]
        assert first.rstrip(".") in {
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"
        }


def test_parent_text_is_the_enclosing_section():
    chunks = split_text(FIXTURE_DOC, "guide.md", chunk_size=200, overlap=0)
    net = next(c for c in chunks if "Networking" in c.breadcrumb)
    assert "Timeout is 30s" in net.parent_text
    assert "exponential backoff" in net.parent_text
    assert "workers" not in net.parent_text  # that is the Configuration section


def test_content_hash_is_stable():
    a = split_text(FIXTURE_DOC, "guide.md", chunk_size=600, overlap=0)
    b = split_text(FIXTURE_DOC, "guide.md", chunk_size=600, overlap=0)
    assert [c.content_hash for c in a] == [c.content_hash for c in b]


def test_empty_and_blank_documents():
    assert split_text("", "empty.md") == []
    assert split_text("   \n\n  ", "blank.md") == []


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("text", "x.md", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("text", "x.md", chunk_size=100, overlap=100)
