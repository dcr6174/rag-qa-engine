## Summary

<!-- What changed and why. Link an issue if there is one. -->

## Test plan

- [ ] `pytest tests/ -q`
- [ ] `python scripts/run_eval.py` (paste before/after metrics if retrieval-relevant)
- [ ] `ruff check src/ tests/ scripts/`
- [ ] `black --check src/ tests/ scripts/`
- [ ] `mypy src/rag_qa/`

## Eval impact

<!--
If this touches chunking, retrieval, reranking, abstention, or generation,
note what moved in `python scripts/run_eval.py`'s output (hit@k, MRR,
nDCG@k, faithfulness, abstention rate, false-answer rate) and why.
Otherwise delete this section.
-->
