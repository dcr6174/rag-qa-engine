#!/usr/bin/env python3
"""Run the evaluation harness: python scripts/run_eval.py"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_qa.evaluate import evaluate, load_qa_pairs
from rag_qa.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pipeline = RAGPipeline()
    chunks = pipeline.ingest_paths([ROOT / "sample_data"])
    print(f"indexed {chunks} chunks from sample_data/")
    report = evaluate(pipeline, load_qa_pairs(ROOT / "eval" / "qa_pairs.jsonl"))
    print(report.summary())
    for detail in report.details:
        flag = "HIT " if detail["hit"] else "MISS"
        print(f"[{flag}] {detail['question']} (rank={detail['rank']}, faithfulness={detail['faithfulness']})")


if __name__ == "__main__":
    main()
