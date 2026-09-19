# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"
```

This installs the package in editable mode plus `pytest`, `httpx`, `ruff`,
`black`, and `mypy`. For exact, pinned dependency versions instead, use
`pip install -r requirements-lock.txt -e .`.

## Before opening a PR

```bash
pytest tests/ -q                        # unit + API tests
python scripts/run_eval.py              # retrieval regression on the fixture corpus
ruff check src/ tests/ scripts/
black --check src/ tests/ scripts/
mypy src/rag_qa/
```

All five run in CI; a PR that fails any of them won't be merged. `black`
autoformats in place if you run it without `--check`.

## What matters most in a review here

This project's whole point is that it's honest about retrieval quality, so:

- **A change that improves one metric at the cost of another needs a reason.**
  `python scripts/run_eval.py` prints hit@k, MRR, nDCG@k, faithfulness,
  abstention rate, and false-answer rate on the bundled fixture corpus - say
  in the PR description what moved and why.
- **Citation offsets must never drift.** If you touch `chunking.py` or
  `citations.py`, the citation-offset property test in `tests/test_citations.py`
  is the one that matters most; don't weaken it to make a change pass.
- **Keep the offline path offline.** `sentence-transformers` and `openai` are
  optional extras (`pip install -e ".[neural]"` / `.[llm]"`); the default
  `hashing` embedder and `lexical` reranker must keep working with zero
  network access and zero extra dependencies.
- **New endpoints or filesystem access need a security look.** See the
  README's "Security notes" - anything that reads or writes an arbitrary
  caller-supplied path needs to go through (or extend) `_confine()` in
  `api.py`, not bypass it.

## Style

Formatting and import order are enforced by `black` and `ruff` (line length
110, see `pyproject.toml`) - don't hand-format around them. Prefer adding a
unit test over a docstring explaining behavior; the codebase leans on tests
and short comments that explain *why*, not *what*.

## Changelog

User-facing changes (new endpoints, behavior changes, bug fixes) should get
an entry in `CHANGELOG.md` under an `[Unreleased]` heading (add one if it
doesn't exist yet).
