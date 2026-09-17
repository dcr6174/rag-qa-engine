# rag-qa-engine

A small, honest retrieval-augmented generation (RAG) app in Python. Add Markdown or text documents in a minimal browser interface, ask questions, and get answers with the source passages behind them.

The project runs **fully offline by default**. Its deterministic hashing embedder and extractive answer generator need no downloads, API keys, or network connection after installation.

![Python](https://img.shields.io/badge/Python-3.10%2B-276749) ![Tests](https://img.shields.io/badge/tests-18%20passing-276749) ![License](https://img.shields.io/badge/license-MIT-276749)

## Use the browser app on Windows

### 1. Install Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). On the installer screen, tick **Add python.exe to PATH**.

### 2. Download this project

Click **Code** near the top of this GitHub page, choose **Download ZIP**, and unzip it. Open the unzipped `rag-qa-engine` folder, click the address bar in File Explorer, type `powershell`, and press Enter.

### 3. Install it (one time)

Run these commands one at a time:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the app

```powershell
uvicorn rag_qa.api:app --app-dir src
```

Leave PowerShell open. Visit **http://127.0.0.1:8000** in Chrome or Edge.

### 5. Ask your documents

1. Drop `.md` or `.txt` files into the upload area, or click it to choose files.
2. Wait until the page says the files are ready.
3. Type a question and click **Ask**.
4. Read the answer and its source cards. Each card names the source file, shows its retrieval match, and includes the exact supporting passage.

The browser sends document text only to the app running on your own computer. The default setup does not upload it to an external service. Stop the app with `Ctrl+C` in PowerShell.

## Mac or Linux setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn rag_qa.api:app --app-dir src
```

Then open http://127.0.0.1:8000.

## API

The existing API remains available at http://127.0.0.1:8000/docs.

```bash
# Index the bundled sample_data folder
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'

# Ask a question
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How does RAG reduce hallucination?"}'
```

Endpoints:

- `GET /` - browser interface
- `GET /health` - service and index status
- `POST /ingest` - index `.md` and `.txt` files from server-side paths
- `POST /ingest-text` - index browser-supplied plain-text documents without saving uploads
- `POST /ask` - answer with ranked source passages

## Architecture

```text
.md / .txt documents
        |
        v
  chunking.py       paragraph-aware splitting with overlap
        |
        v
  embeddings.py     offline HashingEmbedder (or sentence-transformers)
        |
        v
  store.py          in-memory normalized vectors + cosine search
        |
        v
  pipeline.py       retrieve passages -> grounded answer + citations
        |
        +---- api.py + static/index.html (FastAPI API and browser UI)
        +---- evaluate.py (hit@k, MRR, faithfulness)
```

Design choices:

- **Cited answers.** Every answer includes the passages behind it.
- **Private by default.** Browser-selected files are read as text, held in memory, and not written to disk by the upload endpoint.
- **Provider-agnostic LLM layer.** `OpenAICompatibleGenerator` can use an OpenAI-compatible endpoint configured through environment variables. Secrets are never written to the repository.
- **Offline-first testing.** Deterministic embeddings keep tests fast and reproducible.

## Tests and evaluation

With the virtual environment active:

**Windows PowerShell**

```powershell
$env:PYTHONPATH="src"
pytest tests/ -q
python scripts/run_eval.py
```

**Mac or Linux**

```bash
PYTHONPATH=src pytest tests/ -q
python scripts/run_eval.py
```

The 18 tests cover chunking, embeddings, retrieval, cited answers, evaluation, the browser page, browser document ingestion, validation, and the empty-index error. The bundled evaluation set reports hit@4, mean reciprocal rank, and a transparent lexical faithfulness score.

## Optional higher-quality models

The app does not need these to work.

```bash
pip install sentence-transformers
pip install openai
```

Set environment variables in your own terminal, never in committed files:

```bash
export RAG_EMBEDDER=sentence-transformers
export OPENAI_API_KEY=your-key
export RAG_MODEL=gpt-4o-mini
```

## Project layout

```text
src/rag_qa/
  api.py             API endpoints and browser app entry point
  static/index.html  minimal upload, question, answer, and citation UI
  chunking.py        paragraph-aware splitting
  embeddings.py      offline hashing + optional neural embeddings
  store.py           in-memory vector store
  pipeline.py        retrieval and grounded answer generation
  evaluate.py        retrieval and faithfulness metrics
scripts/run_eval.py
sample_data/          bundled demo documents
eval/qa_pairs.jsonl   labelled evaluation questions
tests/                18 automated tests
```

## Common errors

- **`python is not recognized`** - reinstall Python and tick **Add python.exe to PATH**.
- **PowerShell blocks activation** - run `Set-ExecutionPolicy -Scope Process Bypass`, then `.venv\Scripts\activate` again.
- **`No module named uvicorn`** - activate `.venv`, then run `pip install -r requirements.txt`.
- **Port 8000 is already in use** - start with `uvicorn rag_qa.api:app --app-dir src --port 8001`, then open http://127.0.0.1:8001.
- **The app says to add documents first** - upload at least one non-empty `.md` or `.txt` file before asking.
- **You selected a PDF or Word file** - this version accepts plain text and Markdown only. Save or export the document as `.txt` first.

## License

MIT - see [LICENSE](LICENSE).
