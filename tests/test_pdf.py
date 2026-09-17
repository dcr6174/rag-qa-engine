from rag_qa.pdf import extract_pages_from_bytes
from conftest import make_pdf


def test_pages_are_extracted_with_numbers():
    pages = extract_pages_from_bytes(make_pdf(["Timeout is 30 seconds", "Second page about retries"]))
    assert [p.page for p in pages] == [1, 2]
    assert "Timeout" in pages[0].text
    assert "retries" in pages[1].text


def test_empty_pages_are_skipped():
    pages = extract_pages_from_bytes(make_pdf(["Content here", ""]))
    assert len(pages) == 1
