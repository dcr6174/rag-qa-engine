from rag_qa.rerank import LexicalReranker, band_for_score, get_reranker


def test_bands_are_coarse_and_ordered():
    assert band_for_score(0.9) == "high"
    assert band_for_score(0.5) == "medium"
    assert band_for_score(0.05) == "low"


def test_relevant_text_scores_above_irrelevant():
    r = LexicalReranker()
    scores = r.score_pairs(
        "networking timeout configuration",
        ["timeout is 30s in networking configuration", "the sky is blue and nice"],
    )
    assert scores[0] > scores[1]


def test_absent_distinctive_terms_score_near_zero():
    # this is what makes abstention possible: distinctive terms missing
    # everywhere must not accumulate score through shared stopwords
    r = LexicalReranker()
    scores = r.score_pairs(
        "airspeed velocity unladen swallow",
        ["a passage about deployment configuration and timeouts"] * 4,
    )
    assert max(scores) < 0.05


def test_factory_defaults_to_lexical():
    assert get_reranker().kind == "lexical"
    assert get_reranker(None).kind == "lexical"
