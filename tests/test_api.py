from fastapi.testclient import TestClient

from rag_qa import api
from rag_qa.pipeline import RAGPipeline


def client_with_empty_index(monkeypatch) -> TestClient:
    monkeypatch.setattr(api, "pipeline", RAGPipeline(embedder_kind="hashing"))
    return TestClient(api.app)


def test_home_renders_beginner_interface(monkeypatch):
    client = client_with_empty_index(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert "Answers grounded in your words" in response.text
    assert 'id="fileInput"' in response.text
    assert 'id="question"' in response.text


def test_browser_ingest_then_ask_returns_citations(monkeypatch):
    client = client_with_empty_index(monkeypatch)
    ingest = client.post(
        "/ingest-text",
        json={"documents": [{"name": "policy.md", "text": "Release approval requires two reviewers."}]},
    )
    assert ingest.status_code == 200
    assert ingest.json()["files"] == ["policy.md"]

    answer = client.post("/ask", json={"question": "What does release approval require?"})
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["sources"][0]["source"] == "policy.md"
    assert "two reviewers" in payload["answer"].lower()


def test_browser_ingest_rejects_unsupported_files(monkeypatch):
    client = client_with_empty_index(monkeypatch)
    response = client.post(
        "/ingest-text",
        json={"documents": [{"name": "resume.pdf", "text": "content"}]},
    )
    assert response.status_code == 400
    assert "only .md and .txt" in response.json()["detail"]


def test_ask_requires_documents(monkeypatch):
    client = client_with_empty_index(monkeypatch)
    response = client.post("/ask", json={"question": "What is indexed?"})
    assert response.status_code == 400
    assert "add documents first" in response.json()["detail"]
