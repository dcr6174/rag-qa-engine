#!/usr/bin/env python3
"""Generate a QA set from your own documents: python scripts/generate_eval.py [docs_dir] [out.jsonl]

The generated set evaluates retrieval on *your* corpus, not on BEIR.
Review the questions before trusting the numbers - the offline generator
writes keyword-style questions.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_qa.evaluate import generate_qa_pairs
from rag_qa.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_data"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "eval" / "generated_qa_pairs.jsonl"
    pipeline = RAGPipeline()
    pipeline.ingest_paths([docs_dir])
    pairs = generate_qa_pairs(pipeline)
    if not pairs:
        print("no QA pairs could be generated from this corpus")
        raise SystemExit(1)
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} QA pairs to {out}")
    print("tip: add a few {\"question\": ..., \"expected_sources\": [], \"unanswerable\": true}")
    print("rows to measure the false-answer rate on your corpus")


if __name__ == "__main__":
    main()
