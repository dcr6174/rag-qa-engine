"""FastAPI service and beginner-friendly browser interface.

Run:  uvicorn rag_qa.api:app --app-dir src --reload
App:   http://127.0.0.1:8000
Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .citations import ABSTENTION_THRESHOLD
from .evaluate import evaluate, generate_qa_pairs, load_qa_pairs, sweep
from .pipeline import RAGPipeline

app = FastAPI(title="rag-qa-engine", version="0.3.0")
pipeline = RAGPipeline()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = PROJECT_ROOT / "sample_data"
DEFAULT_QA = PROJECT_ROOT / "eval" / "qa_pairs.jsonl"
INDEX_HTML = Path(__file__).with_name("static") / "index.html"

# Short in-process conversation history for follow-up query rewriting.
_history: list[dict] = []


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: [str(DEFAULT_DOCS)])


class TextDocument(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=2_000_000)


class IngestTextRequest(BaseModel):
    documents: list[TextDocument] = Field(min_length=1, max_length=20)


class PdfDocument(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    data_base64: str = Field(min_length=1)


class IngestPdfRequest(BaseModel):
    documents: list[PdfDocument] = Field(min_length=1, max_length=5)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)


class EvalRequest(BaseModel):
    qa_path: str | None = None
    generate: bool = False
    max_generated: int = Field(default=12, ge=1, le=50)
    k: int = Field(default=4, ge=1, le=10)


class SweepRequest(BaseModel):
    qa_path: str | None = None
    max_configs: int = Field(default=6, ge=1, le=12)
    k: int = Field(default=4, ge=1, le=10)


class StoreRequest(BaseModel):
    directory: str = Field(min_length=1, max_length=500)


def _source_out(r) -> dict:
    return {
        "source": r.source,
        "chunk_index": r.metadata.get("chunk_index"),
        "rank": r.final_rank,
        "band": r.band,
        "page": r.metadata.get("page"),
        "breadcrumb": r.metadata.get("breadcrumb", ""),
        "text": r.text,
    }


def _answer_payload(result) -> dict:
    return {
        "answer": result.text,
        "abstained": result.abstained,
        "rewritten_question": result.rewritten_question,
        "unsupported_sentences": result.unsupported_sentences,
        "citations": [c.to_dict() for c in result.citations],
        "sources": [_source_out(s) for s in result.sources],
        "trace": result.trace,
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "indexed_chunks": len(pipeline.store),
        "abstention_threshold": pipeline.abstention_threshold,
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    added = pipeline.ingest_paths(request.paths)
    if added == 0 and len(pipeline.store) == 0:
        raise HTTPException(status_code=400, detail="no supported documents found at the given paths")
    return {"chunks_embedded": added, "indexed_chunks": len(pipeline.store)}


@app.post("/ingest-text")
def ingest_text(request: IngestTextRequest) -> dict:
    """Index plain-text documents sent by the browser without storing uploads."""
    embedded = 0
    accepted: list[str] = []
    for document in request.documents:
        name = Path(document.name).name
        if Path(name).suffix.lower() not in {".md", ".txt"}:
            raise HTTPException(status_code=400, detail=f"{name}: only .md and .txt files are supported")
        chunks = pipeline.ingest_text(document.text, name)
        accepted.append(name)
        embedded += chunks
    return {
        "files": accepted,
        "chunks_embedded": embedded,
        "indexed_chunks": len(pipeline.store),
    }


@app.post("/ingest-pdf")
def ingest_pdf(request: IngestPdfRequest) -> dict:
    """Index PDF uploads with page-anchored citations. Uploads are not stored."""
    from .pdf import extract_pages_from_bytes

    accepted: list[str] = []
    embedded = 0
    for document in request.documents:
        name = Path(document.name).name
        if Path(name).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"{name}: only .pdf files are supported here")
        try:
            data = base64.b64decode(document.data_base64)
            pages = extract_pages_from_bytes(data)
        except Exception as exc:  # corrupt PDF, encrypted PDF, bad base64
            raise HTTPException(status_code=400, detail=f"{name}: could not read PDF ({exc})") from exc
        if not pages:
            raise HTTPException(
                status_code=400,
                detail=f"{name}: no text layer found - scanned PDFs are not supported",
            )
        embedded += pipeline.ingest_pdf_pages(pages, name)
        accepted.append(name)
    return {"files": accepted, "chunks_embedded": embedded, "indexed_chunks": len(pipeline.store)}


def _require_index() -> None:
    if len(pipeline.store) == 0:
        raise HTTPException(status_code=400, detail="index is empty - add documents first")


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    _require_index()
    result = pipeline.answer(request.question, request.k, history=_history[-6:])
    _history.append({"question": request.question})
    del _history[:-12]
    return _answer_payload(result)


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: sources first, then the answer, then citations.

    Retrieval finishes before generation, so sources can appear immediately
    while the answer types out - and the abstention state reaches the screen
    without waiting for a generator at all.
    """
    _require_index()
    result = pipeline.answer(request.question, request.k, history=_history[-6:])
    _history.append({"question": request.question})
    del _history[:-12]

    def events():
        yield f"event: sources\ndata: {json.dumps({'sources': [_source_out(s) for s in result.sources], 'abstained': result.abstained, 'rewritten_question': result.rewritten_question})}\n\n"
        if not result.abstained:
            for citation in result.citations:
                # stream the answer one sentence at a time, in order
                yield f"event: sentence\ndata: {json.dumps({'text': citation.sentence, 'entailed': citation.entailed})}\n\n"
        yield f"event: done\ndata: {json.dumps(_answer_payload(result))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/evaluate")
def run_evaluation(request: EvalRequest) -> dict:
    """Evaluate retrieval on a labelled QA set, or generate one from the corpus."""
    _require_index()
    if request.generate or request.qa_path is None:
        qa_pairs = generate_qa_pairs(pipeline, request.max_generated)
        if not qa_pairs:
            raise HTTPException(status_code=400, detail="could not generate QA pairs from this corpus")
    else:
        qa_path = Path(request.qa_path)
        if not qa_path.exists():
            raise HTTPException(status_code=400, detail=f"QA file not found: {qa_path}")
        qa_pairs = load_qa_pairs(qa_path)
    report = evaluate(pipeline, qa_pairs, k=request.k)
    return {
        "questions": report.questions,
        "hit_at_k": round(report.hit_at_k, 3),
        "mrr": round(report.mrr, 3),
        "ndcg_at_k": round(report.ndcg_at_k, 3),
        "recall_at_k": {str(k): round(v, 3) for k, v in report.recall_at_k.items()},
        "faithfulness": round(report.faithfulness, 3),
        "false_answer_rate": report.false_answer_rate,
        "abstention_rate": round(report.abstention_rate, 3),
        "generated": request.generate or request.qa_path is None,
        "qa_pairs": qa_pairs,
        "details": report.details,
    }


@app.post("/sweep")
def run_sweep(request: SweepRequest) -> dict:
    """Config sweep over the indexed corpus: chunk size, overlap, k."""
    _require_index()
    qa_path = Path(request.qa_path) if request.qa_path else DEFAULT_QA
    qa_pairs = load_qa_pairs(qa_path) if qa_path.exists() else generate_qa_pairs(pipeline, 12)
    documents = dict(pipeline.documents)
    leaderboard = sweep(documents, qa_pairs, k=request.k, max_configs=request.max_configs)
    return {"configs": len(leaderboard), "leaderboard": leaderboard, "questions": len(qa_pairs)}


@app.post("/store/save")
def store_save(request: StoreRequest) -> dict:
    _require_index()
    pipeline.save_store(request.directory)
    return {"saved": request.directory, "indexed_chunks": len(pipeline.store), "meta": pipeline.store_meta()}


@app.post("/store/load")
def store_load(request: StoreRequest) -> dict:
    from .store import StoreMismatchError

    try:
        pipeline.load_store(request.directory)
    except StoreMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no store found at {request.directory}") from exc
    return {"loaded": request.directory, "indexed_chunks": len(pipeline.store)}
