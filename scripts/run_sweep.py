#!/usr/bin/env python3
"""Sweep retrieval configs over sample_data and print the leaderboard.

Usage: python scripts/run_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_qa.evaluate import load_qa_pairs, sweep

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = {f.name: f.read_text(encoding="utf-8") for f in sorted((ROOT / "sample_data").glob("*.md"))}
    qa_pairs = load_qa_pairs(ROOT / "eval" / "qa_pairs.jsonl")
    leaderboard = sweep(documents, qa_pairs)
    print(
        f"{'chunk':>6} {'overlap':>8} {'k':>3} {'hit@k':>6} {'mrr':>6} "
        f"{'ndcg@k':>7} {'faith':>6} {'false-ans':>10}"
    )
    for row in leaderboard:
        far = "-" if row["false_answer_rate"] is None else f"{row['false_answer_rate']:.2f}"
        print(
            f"{row['chunk_size']:>6} {row['overlap']:>8} {row['top_k']:>3} "
            f"{row['hit_at_k']:>6.2f} {row['mrr']:>6.2f} {row['ndcg_at_k']:>7.2f} "
            f"{row['faithfulness']:>6.2f} {far:>10}"
        )


if __name__ == "__main__":
    main()
