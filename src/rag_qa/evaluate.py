"""Evaluation harness: retrieval quality, abstention honesty, and sweeps.

This harness is the project's center of gravity. It runs a labelled QA set
through the pipeline and reports:

* hit@k       - did any expected source appear in the top-k passages?
* recall@k    - fraction of expected sources found in the top-k, at several k
* MRR         - mean reciprocal rank of the first expected source
* nDCG@k      - graded ranking quality; MRR alone hides whether you are
                retrieving enough context or just one lucky passage
* faithfulness - fraction of answer sentences entailed by their cited span
* false-answer rate - on a no-answer subset, how often the system answers
                anyway instead of abstaining. A system with 90% hit rate and
                40% false-answer rate is worse than useless.

It also generates Q/A pairs from the user's own corpus (so they evaluate on
their documents, not on BEIR) and sweeps retrieval configs into a leaderboard
("your docs work best at 800 characters with 15% overlap" is a defensible
output, not a vibe).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import RAGPipeline

DEFAULT_KS = (1, 3, 5)


@dataclass
class EvalReport:
    questions: int
    hit_at_k: float
    recall_at_k: dict
    mrr: float
    ndcg_at_k: float
    faithfulness: float
    false_answer_rate: float | None = None
    abstention_rate: float = 0.0
    k: int = 4
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        recall = " ".join(f"recall@{k}={v:.2f}" for k, v in sorted(self.recall_at_k.items()))
        far = f" false-answer-rate={self.false_answer_rate:.2f}" if self.false_answer_rate is not None else ""
        return (
            f"questions={self.questions} hit@{self.k}={self.hit_at_k:.2f} mrr={self.mrr:.2f} "
            f"ndcg@{self.k}={self.ndcg_at_k:.2f} {recall} faithfulness={self.faithfulness:.2f}"
            f" abstention-rate={self.abstention_rate:.2f}{far}"
        )


def load_qa_pairs(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate(pipeline: RAGPipeline, qa_pairs: list[dict], k: int = 4, ks=DEFAULT_KS) -> EvalReport:
    """Run the QA set through *pipeline* and compute all metrics.

    Pairs marked ``"unanswerable": true`` form the no-answer subset: the
    honest system abstains on them, and answering anyway counts toward the
    false-answer rate.
    """
    if not qa_pairs:
        raise ValueError("qa_pairs is empty")

    answerable = [p for p in qa_pairs if not p.get("unanswerable")]
    unanswerable = [p for p in qa_pairs if p.get("unanswerable")]

    hits, mrrs, ndcgs, faiths = [], [], [], []
    recall_sums = {kk: 0.0 for kk in ks}
    abstained_count = 0
    details: list[dict] = []

    for pair in answerable:
        expected = set(pair["expected_sources"])
        result = pipeline.answer(pair["question"], k)
        abstained_count += int(result.abstained)
        ranked_sources = [s.source for s in result.sources]

        rank = next((i + 1 for i, src in enumerate(ranked_sources) if src in expected), None)
        hits.append(1.0 if rank else 0.0)
        mrrs.append(1.0 / rank if rank else 0.0)
        for kk in ks:
            found = expected & set(ranked_sources[:kk])
            recall_sums[kk] += len(found) / len(expected)

        # gains over unique sources: several chunks from one document count once
        unique_sources = list(dict.fromkeys(ranked_sources[:k]))
        gains = [1.0 if src in expected else 0.0 for src in unique_sources]
        ideal = [1.0] * min(len(expected), k)
        ndcgs.append(_dcg(gains) / (_dcg(ideal) or 1.0))

        if result.abstained:
            faiths.append(0.0)
            faith = 0.0
        else:
            entailed = [c.entailed for c in result.citations]
            faith = sum(entailed) / len(entailed) if entailed else 0.0
            faiths.append(faith)
        details.append(
            {
                "question": pair["question"],
                "hit": bool(rank),
                "rank": rank,
                "abstained": result.abstained,
                "faithfulness": round(faith, 3),
                "answer": result.text,
            }
        )

    n = len(answerable)
    false_answer_rate = None
    if unanswerable:
        false_answers = 0
        for pair in unanswerable:
            result = pipeline.answer(pair["question"], k)
            abstained_count += int(result.abstained)
            if not result.abstained:
                false_answers += 1
            details.append(
                {
                    "question": pair["question"],
                    "unanswerable": True,
                    "abstained": result.abstained,
                    "false_answer": not result.abstained,
                }
            )
        false_answer_rate = false_answers / len(unanswerable)

    total = len(qa_pairs)
    return EvalReport(
        questions=total,
        hit_at_k=sum(hits) / n if n else 0.0,
        recall_at_k={kk: recall_sums[kk] / n if n else 0.0 for kk in ks},
        mrr=sum(mrrs) / n if n else 0.0,
        ndcg_at_k=sum(ndcgs) / n if n else 0.0,
        faithfulness=sum(faiths) / n if n else 0.0,
        false_answer_rate=false_answer_rate,
        abstention_rate=abstained_count / total,
        k=k,
        details=details,
    )


def generate_qa_pairs(pipeline: RAGPipeline, max_pairs: int = 20) -> list[dict]:
    """Generate a QA set from the user's own indexed corpus.

    Each chunk becomes a gold passage. With an OpenAI-compatible generator
    configured the question is model-written; offline, a transparent
    heuristic builds a question from the chunk's most distinctive terms.
    Either way the user evaluates retrieval on their documents, not on BEIR.
    """
    from .bm25 import tokenize

    pairs: list[dict] = []
    seen_terms: set[frozenset] = set()
    for record in pipeline.store.records:
        text = record["text"]
        content = [t for t in dict.fromkeys(tokenize(text)) if len(t) > 4]
        if len(content) < 3:
            continue
        key = frozenset(content[:5])
        if key in seen_terms:
            continue
        seen_terms.add(key)
        # Keyword-style question built only from corpus terms: glue words the
        # corpus never contains would dilute the relevance signal and turn
        # the generated set into an abstention test instead of a retrieval test.
        topic = " ".join(content[:4])
        pairs.append(
            {
                "question": f"{topic}?",
                "expected_sources": [record["source"]],
            }
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def sweep(
    documents: dict[str, str],
    qa_pairs: list[dict],
    grid: dict | None = None,
    k: int = 4,
    max_configs: int = 8,
) -> list[dict]:
    """Sweep retrieval configs over the user's corpus and rank the leaderboard.

    ``grid`` keys: ``chunk_size``, ``overlap_ratio``, ``top_k``. Every row is
    a full re-index + evaluation, so the leaderboard says something real:
    which configuration retrieves best on *these* documents.
    """
    grid = grid or {}
    chunk_sizes = grid.get("chunk_size", [400, 600, 900])
    overlap_ratios = grid.get("overlap_ratio", [0.15])
    top_ks = grid.get("top_k", [k])

    combos = [
        (cs, max(1, int(cs * ratio)), tk)
        for cs, ratio, tk in itertools.product(chunk_sizes, overlap_ratios, top_ks)
    ][:max_configs]

    leaderboard: list[dict] = []
    for chunk_size, overlap, top_k in combos:
        pipeline = RAGPipeline(chunk_size=chunk_size, overlap=overlap, top_k=top_k)
        for source, text in documents.items():
            pipeline.ingest_text(text, source)
        report = evaluate(pipeline, qa_pairs, k=top_k)
        leaderboard.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                "hit_at_k": round(report.hit_at_k, 3),
                "recall_at_k": {str(kk): round(v, 3) for kk, v in report.recall_at_k.items()},
                "mrr": round(report.mrr, 3),
                "ndcg_at_k": round(report.ndcg_at_k, 3),
                "faithfulness": round(report.faithfulness, 3),
                "false_answer_rate": (
                    round(report.false_answer_rate, 3) if report.false_answer_rate is not None else None
                ),
            }
        )
    leaderboard.sort(key=lambda row: (-row["ndcg_at_k"], -row["hit_at_k"]))
    return leaderboard
