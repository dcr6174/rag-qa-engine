"""Eval-as-regression-test: the fixture corpus must stay above metric floors.

This is the test that catches "we changed the chunker and retrieval quietly
got worse" - the eval harness runs in CI like any other test.
"""

from rag_qa.evaluate import evaluate, load_qa_pairs
from rag_qa.pipeline import RAGPipeline

HIT_FLOOR = 0.90
MRR_FLOOR = 0.80
FALSE_ANSWER_CEILING = 0.10
FAITHFULNESS_FLOOR = 0.70


def test_fixture_corpus_stays_above_metric_floors():
    pipeline = RAGPipeline()
    pipeline.ingest_paths(["sample_data"])
    report = evaluate(pipeline, load_qa_pairs("eval/qa_pairs.jsonl"))
    assert report.hit_at_k >= HIT_FLOOR, report.summary()
    assert report.mrr >= MRR_FLOOR, report.summary()
    assert report.faithfulness >= FAITHFULNESS_FLOOR, report.summary()
    assert report.false_answer_rate is not None
    assert report.false_answer_rate <= FALSE_ANSWER_CEILING, report.summary()
