"""FastAPI service and beginner-friendly browser interface.

Run:  uvicorn rag_qa.api:app --app-dir src --reload
App:   http://127.0.0.1:8000
Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

app = FastAPI(title="rag-qa-engine", version="0.2.0")
pipeline = RAGPipeline()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = PROJECT_ROOT / "sample_data"
INDEX_HTML = Path(__file__).with_name("static") / "index.html"


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: [str(DEFAULT_DOCS)])


class TextDocument(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=2_000_000)


class IngestTextRequest(BaseModel):
    documents: list[TextDocument] = Field(min_length=1, max_length=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)


class SourceOut(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(pipeline.store)}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    added = pipeline.ingest_paths(request.paths)
    if added == 0:
        raise HTTPException(status_code=400, detail="no .md/.txt documents found at the given paths")
    return {"chunks_added": added, "indexed_chunks": len(pipeline.store)}


@app.post("/ingest-text")
def ingest_text(request: IngestTextRequest) -> dict:
    """Index plain-text documents sent by the browser without storing uploads."""
    added = 0
    accepted: list[str] = []
    for document in request.documents:
        name = Path(document.name).name
        if Path(name).suffix.lower() not in {".md", ".txt"}:
            raise HTTPException(status_code=400, detail=f"{name}: only .md and .txt files are supported")
        chunks = pipeline.ingest_text(document.text, name)
        if chunks:
            added += chunks
            accepted.append(name)
    if added == 0:
        raise HTTPException(status_code=400, detail="the selected documents were empty")
    return {
        "files": accepted,
        "chunks_added": added,
        "indexed_chunks": len(pipeline.store),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if len(pipeline.store) == 0:
        raise HTTPException(status_code=400, detail="index is empty - add documents first")
    result = pipeline.answer(request.question, request.k)
    return AskResponse(
        answer=result.text,
        sources=[SourceOut(source=s.source, score=round(s.score, 4), text=s.text) for s in result.sources],
    )
