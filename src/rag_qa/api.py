"""FastAPI service exposing the pipeline over HTTP.

Run:  uvicorn rag_qa.api:app --app-dir src --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

app = FastAPI(title="rag-qa-engine", version="0.1.0")
pipeline = RAGPipeline()

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "sample_data"


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: [str(DEFAULT_DOCS)])


class AskRequest(BaseModel):
    question: str
    k: int = 4


class SourceOut(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(pipeline.store)}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    added = pipeline.ingest_paths(request.paths)
    if added == 0:
        raise HTTPException(status_code=400, detail="no .md/.txt documents found at the given paths")
    return {"chunks_added": added, "indexed_chunks": len(pipeline.store)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if len(pipeline.store) == 0:
        raise HTTPException(status_code=400, detail="index is empty - POST /ingest first")
    result = pipeline.answer(request.question, request.k)
    return AskResponse(
        answer=result.text,
        sources=[SourceOut(source=s.source, score=round(s.score, 4), text=s.text) for s in result.sources],
    )
