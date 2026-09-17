"""Rerankers: re-score fused candidates with a finer relevance judgment.

Two interchangeable rerankers, mirroring the embedder design:

* ``LexicalReranker`` - deterministic, offline, zero downloads. Combines
  normalized BM25 score, query-term coverage, and exact-phrase bonus. This is
  the default so the whole project runs without a network connection.
* ``CrossEncoderReranker`` - a trained cross-encoder (default
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``) via the optional
  ``sentence-transformers`` package. Runs on CPU; a trained reranker lifts
  retrieval quality more than any local embedding-model upgrade.

Both return a score in [0, 1]. Scores are turned into coarse bands
(high / medium / low) for display: a raw number invites users to read it as
calibrated confidence in the answer, which it is not. A rank plus a coarse
band from the reranker is the honest presentation.
"""

from __future__ import annotations

import math

from .bm25 import stem, tokenize

# Query stopwords carry no relevance signal but, when absent from every
# candidate, acquire a large idf weight and crush coverage scores - which
# breaks abstention on small corpora. Removed from queries only.
STOPWORDS = {
    stem(w)
    for w in (
        "a an and are as at be been by for from has have had how i in is it its of on or "
        "that the this to was were what when where which who whom why will with you your "
        "we our they their them he she his her do does did can could should would may "
        "might must me my mine us not no if then than so such into about over under "
        "between per via am any all each other more most some"
    ).split()
}

BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"


def band_for_score(score: float) -> str:
    """Coarse relevance band for a reranker score in [0, 1].

    Bands are a trained/heuristic relevance judgment, deliberately coarse:
    they say "how relevant is this passage", never "how confident is the
    answer".
    """
    if score >= 0.65:
        return BAND_HIGH
    if score >= 0.30:
        return BAND_MEDIUM
    return BAND_LOW


class LexicalReranker:
    """Offline reranker: idf-weighted coverage + frequency + phrase bonus.

    Term weights come from the candidate set itself (rare terms - error
    codes, version numbers, proper nouns - weigh most; stopwords weigh
    almost nothing), which is what lets abstention work: a question whose
    distinctive terms appear nowhere scores near zero. Deterministic.
    """

    kind = "lexical"

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        query_terms = {t for t in tokenize(query) if t not in STOPWORDS}
        if not query_terms or not texts:
            return [0.0] * len(texts)
        doc_tokens = [tokenize(t) for t in texts]
        doc_term_sets = [set(t) for t in doc_tokens]
        n = len(texts)
        # Candidate-set idf: a term in every candidate carries no signal.
        df = {t: sum(1 for s in doc_term_sets if t in s) for t in query_terms}
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        total_idf = sum(idf.values()) or 1.0
        phrase = " ".join(tokenize(query))
        scores = []
        for tokens, term_set in zip(doc_tokens, doc_term_sets):
            coverage = sum(idf[t] for t in query_terms & term_set) / total_idf
            freq = sum(tokens.count(t) * idf[t] for t in query_terms) / ((len(tokens) or 1) * total_idf)
            bm25ish = min(1.0, freq * 30)
            phrase_bonus = 1.0 if len(phrase) > 3 and phrase in " ".join(tokens) else 0.0
            score = 0.60 * coverage + 0.25 * bm25ish + 0.15 * phrase_bonus
            scores.append(min(1.0, score))
        return scores


class CrossEncoderReranker:
    """Trained cross-encoder reranker (optional sentence-transformers dep)."""

    kind = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # imported lazily

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        raw = self._model.predict([(query, t) for t in texts])
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]  # sigmoid to [0,1]


def get_reranker(kind: str | None = None, **kwargs):
    """Factory: ``lexical`` (default) or ``cross-encoder``."""
    if kind in {None, "lexical"}:
        return LexicalReranker(**kwargs)
    if kind in {"cross-encoder", "ce"}:
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"unknown reranker kind: {kind!r}")
