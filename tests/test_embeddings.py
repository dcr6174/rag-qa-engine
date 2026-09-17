import numpy as np

from rag_qa.embeddings import HashingEmbedder


def test_embeddings_are_normalized_and_deterministic():
    embedder = HashingEmbedder()
    first = embedder.embed(["retrieval augmented generation"])
    second = embedder.embed(["retrieval augmented generation"])
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_similar_texts_score_higher_than_unrelated():
    embedder = HashingEmbedder()
    a, b, c = embedder.embed(
        ["vector search over document chunks",
         "document chunks ranked by vector search",
         "banana bread recipe with walnuts"]
    )
    assert float(a @ b) > float(a @ c)
