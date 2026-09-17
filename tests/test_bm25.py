from rag_qa.bm25 import BM25Index, stem, tokenize


def test_keyword_match_beats_unrelated():
    idx = BM25Index()
    idx.add(["error E42 indicates a timeout", "the sky is blue", "unrelated prose entirely"])
    hits = idx.search("E42 error")
    assert hits[0][0] == 0


def test_rare_terms_outrank_common_terms():
    idx = BM25Index()
    idx.add(["timeout timeout timeout", "timeout and SKU-9917", "nothing here"])
    hits = dict(idx.search("SKU-9917", k=3))
    assert hits[1] > hits[0]


def test_empty_index_returns_no_hits():
    assert BM25Index().search("anything") == []


def test_stemming_matches_morphological_variants():
    assert "reduce" in tokenize("reduces")
    assert "metric" in tokenize("metrics")
    assert "configure" in tokenize("configured") or "configur" in tokenize("configured")
    assert "this" in tokenize("this")  # conservative: stopwords survive


def test_stem_rules():
    assert stem("running") == "runn"
    assert stem("files") == "file"
    assert stem("is") == "is"
    assert stem("bus") == "bus"
