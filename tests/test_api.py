import base64

import pytest
from fastapi.testclient import TestClient

from rag_qa.api import app, pipeline
from conftest import make_pdf

client = TestClient(client_base := None) if False else TestClient(app)


@pytest.fixture(autouse=True)
def fresh_index():
    from rag_qa import api

    pipeline.store = type(pipeline.store)()
    pipeline.retriever = type(pipeline.retriever)(pipeline.store, pipeline.reranker)
    pipeline.documents = {}
    api._history.clear()
    yield


def _index_samples():
    return client.post("/ingest", json={})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "abstention_threshold" in response.json()


def test_ingest_and_ask_with_full_payload():
    assert _index_samples().status_code == 200
    response = client.post("/ask", json={"question": "How does RAG reduce hallucination?"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["citations"]
    assert data["trace"]
    first = data["sources"][0]
    assert {"rank", "band", "source", "text"} <= set(first)
    for citation in data["citations"]:
        assert {"sentence", "quote", "doc_start", "doc_end", "entailed", "band"} <= set(citation)


def test_ask_abstention_payload():
    _index_samples()
    data = client.post("/ask", json={"question": "Who won the 2019 cricket world cup?"}).json()
    assert data["abstained"] is True
    assert data["answer"] == ""
    assert data["sources"]


def test_ask_requires_index():
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 400


def test_ingest_text_roundtrip():
    response = client.post(
        "/ingest-text",
        json={"documents": [{"name": "note.md", "text": "Timeout is 30 seconds by default."}]},
    )
    assert response.status_code == 200
    assert response.json()["indexed_chunks"] >= 1
    bad = client.post("/ingest-text", json={"documents": [{"name": "evil.exe", "text": "x"}]})
    assert bad.status_code == 400


def test_ingest_pdf_with_page_anchors():
    payload = base64.b64encode(make_pdf(["Timeout is 30 seconds"])).decode()
    response = client.post("/ingest-pdf", json={"documents": [{"name": "manual.pdf", "data_base64": payload}]})
    assert response.status_code == 200
    data = client.post("/ask", json={"question": "What is the timeout?"}).json()
    assert data["abstained"] is False
    assert any(c["page"] == 1 for c in data["citations"])


def test_ingest_pdf_rejects_non_pdf():
    payload = base64.b64encode(b"plain text").decode()
    response = client.post("/ingest-pdf", json={"documents": [{"name": "note.txt", "data_base64": payload}]})
    assert response.status_code == 400


def test_sse_stream_sends_sources_first():
    _index_samples()
    with client.stream("POST", "/ask/stream", json={"question": "What is MRR?"}) as response:
        body = response.read().decode()
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event:")]
    assert events[0] == "sources"
    assert events[-1] == "done"
    assert "sentence" in events


def test_evaluate_generated_pairs():
    _index_samples()
    response = client.post("/evaluate", json={"generate": True, "max_generated": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["questions"] >= 1
    assert 0.0 <= data["ndcg_at_k"] <= 1.0


def test_sweep_leaderboard():
    _index_samples()
    response = client.post("/sweep", json={"max_configs": 2})
    assert response.status_code == 200
    assert len(response.json()["leaderboard"]) >= 1


def test_store_save_load_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOWED_ROOTS", str(tmp_path))
    _index_samples()
    save = client.post("/store/save", json={"directory": str(tmp_path / "s")})
    assert save.status_code == 200
    load = client.post("/store/load", json={"directory": str(tmp_path / "s")})
    assert load.status_code == 200
    missing = client.post("/store/load", json={"directory": str(tmp_path / "nope")})
    assert missing.status_code in (404, 409)


def test_store_save_rejects_path_outside_allowed_root(tmp_path):
    # No RAG_ALLOWED_ROOTS set: only the project directory is allowed, so a
    # pytest tmp_path (elsewhere on disk) must be rejected rather than
    # silently written to - the API is not an arbitrary file-write endpoint.
    _index_samples()
    response = client.post("/store/save", json={"directory": str(tmp_path / "s")})
    assert response.status_code == 400


def test_ingest_rejects_path_outside_allowed_root(tmp_path):
    (tmp_path / "note.md").write_text("secret content", encoding="utf-8")
    response = client.post("/ingest", json={"paths": [str(tmp_path)]})
    assert response.status_code == 400
