import pytest

from rag_qa.chunking import split_text


def test_empty_text_yields_no_chunks():
    assert split_text("", "doc.md") == []
    assert split_text("   \n\n ", "doc.md") == []


def test_short_text_is_one_chunk():
    chunks = split_text("A short document.", "doc.md")
    assert len(chunks) == 1
    assert chunks[0].source == "doc.md"
    assert chunks[0].index == 0


def test_long_text_respects_chunk_size_and_indexes_in_order():
    text = "\n\n".join(f"Paragraph {i} with some words in it." for i in range(40))
    chunks = split_text(text, "doc.md", chunk_size=200, overlap=40)
    assert len(chunks) > 2
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 260 for c in chunks)  # size + overlap slack


def test_overlap_carries_tail_into_next_chunk():
    text = "alpha bravo charlie delta. " * 30
    chunks = split_text(text, "doc.md", chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    tail = chunks[0].text[-40:]
    shared = [w for w in tail.split() if w in chunks[1].text]
    assert shared, "expected overlapping words between consecutive chunks"


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        split_text("text", "doc.md", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("text", "doc.md", chunk_size=100, overlap=100)
