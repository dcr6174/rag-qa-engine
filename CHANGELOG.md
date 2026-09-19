# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.1] - 2026-09-19

### Added
- CI (`.github/workflows/ci.yml`): pytest and the eval regression harness on
  Python 3.10-3.12, plus a separate lint job (ruff, black, mypy).
- Packaging via `pyproject.toml` (editable install, `neural`/`llm`/`dev`
  extras); `requirements.txt` remains for the README's copy-paste setup.
- `requirements-lock.txt`: exact pinned versions for reproducible eval numbers.
- `Dockerfile` and `.dockerignore`; see the README's "Docker" section.
- `CONTRIBUTING.md` and GitHub issue/PR templates.
- ruff, black, and mypy configuration in `pyproject.toml`.

### Changed
- `/ingest`, `/store/save`, and `/store/load` now confine caller-supplied
  paths to the project directory by default (`RAG_ALLOWED_ROOTS` extends
  it); previously any filesystem path was accepted.
- Documented the single-process, single-user design of the API (global
  `pipeline`/`_history` state) in code and in a new README "Security notes"
  section.

### Fixed
- Structure-aware chunking: a fenced code block (` ``` `) was closed early by
  an unrelated fence marker of the other kind (`~~~`) appearing as literal
  text inside it, splitting the block. Closing now matches the exact marker
  that opened the block.
- Removed a dead `extract_pages` import in `pipeline.py`.

## [0.3.0] - 2026-09-17

### Added
- Evaluation-first RAG pipeline: structure-aware Markdown/PDF chunking with
  stable offsets, hybrid BM25 + dense retrieval fused with RRF, reranking,
  threshold-based abstention, and sentence-level citations with an
  entailment check before display.
- Content-hash incremental indexing and store hygiene (persisted stores
  refuse to load under mismatched embedder/chunker settings).
- Evaluation harness: recall@k, nDCG@k, MRR, faithfulness, false-answer
  rate, corpus QA generation, and a config-sweep leaderboard.
- FastAPI service and a minimal browser interface with sentence citations,
  relevance bands, a retrieval trace, and an evaluation panel.
- 67 automated tests, including an eval-as-regression gate and a
  citation-offset property test.
