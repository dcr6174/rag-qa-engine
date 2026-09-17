"""Evaluation harness: retrieval quality and answer faithfulness.

Runs a labelled QA set (JSONL: question, expected_sources, answer_keywords)
through the pipeline and reports:

* hit@k      - did any expected source appear in the top-k retrieved passages?
* MRR        - mean reciprocal rank of the first expected source
* faithfulness - fraction of answer sentences whose content words overlap the
  retrieved passages (a transparent lexical proxy, not a neural judge)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import RAGPipeline


@dataclass
class EvalReport:
    questions: int
    hit_at_k: float
    mrr: float
    faithfulness: float
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"questions={self.questions} hit@k={self.hit_at_k:.2f} "
            f"mrr={self.mrr:.2f} faithfulness={self.faithfulness:.2f}"
        )


def load_qa_pairs(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(pipeline: RAGPipeline, qa_pairs: list[dict], k: int = 4) -> EvalReport:
    if not qa_pairs:
        raise ValueError("qa_pairs is empty")
    hits, reciprocal_ranks, faithfulness_scores, details = [], [], [], []
    for pair in qa_pairs:
        expected = set(pair["expected_sources"])
        result = pipeline.answer(pair["question"], k)
        ranked_sources = [s.source for s in result.sources]

        rank = next((i + 1 for i, src in enumerate(ranked_sources) if src in expected), None)
        hits.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        faith = _faithfulness(result.text, [s.text for s in result.sources])
        faithfulness_scores.append(faith)
        details.append(
            {
                "question": pair["question"],
                "hit": bool(rank),
                "rank": rank,
                "faithfulness": round(faith, 3),
                "answer": result.text,
            }
        )
    n = len(qa_pairs)
    return EvalReport(
        questions=n,
        hit_at_k=sum(hits) / n,
        mrr=sum(reciprocal_ranks) / n,
        faithfulness=sum(faithfulness_scores) / n,
        details=details,
    )


def _faithfulness(answer: str, passages: list[str]) -> float:
    import re

    passage_words = set(re.findall(r"[a-z0-9]+", " ".join(passages).lower()))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        words = [w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if len(w) > 3]
        if not words:
            supported += 1  # citation-only or numeric fragments carry no claim
            continue
        overlap = sum(1 for w in words if w in passage_words) / len(words)
        if overlap >= 0.5:
            supported += 1
    return supported / len(sentences)
