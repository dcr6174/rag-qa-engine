import numpy as np
import pytest

from rag_qa.embeddings import HashingEmbedder
from rag_qa.store import StoreMismatchError, VectorStore


def _store_with(texts):
    store = VectorStore()
    vectors = HashingEmbedder().embed(texts)
    records = [
        {"text": t, "source": f"doc{i}.md", "metadata": {"content_hash": f"h{i}"}}
        for i, t in enumerate(texts)
    ]
    store.add(vectors, records)
    return store


def test_save_load_roundtrip(tmp_path):
    store = _store_with(["alpha", "beta"])
    meta = {"embedder": "HashingEmbedder", "embedding_dim": 384, "chunker": "markdown-v2:600:100"}
    store.save(tmp_path, meta=meta)
    loaded = VectorStore.load(tmp_path, expected_meta=meta)
    assert len(loaded) == 2
    assert loaded.records[0]["text"] == "alpha"


def test_load_refuses_mismatched_settings(tmp_path):
    store = _store_with(["alpha"])
    store.save(tmp_path, meta={"embedder": "HashingEmbedder", "embedding_dim": 384})
    with pytest.raises(StoreMismatchError, match="embedding_dim"):
        VectorStore.load(tmp_path, expected_meta={"embedder": "HashingEmbedder", "embedding_dim": 768})
    with pytest.raises(StoreMismatchError, match="embedder"):
        VectorStore.load(
            tmp_path, expected_meta={"embedder": "SentenceTransformerEmbedder", "embedding_dim": 384}
        )


def test_load_refuses_store_without_metadata(tmp_path):
    store = _store_with(["alpha"])
    store.save(tmp_path)
    (tmp_path / "store_meta.json").unlink()
    with pytest.raises(StoreMismatchError):
        VectorStore.load(tmp_path, expected_meta={"embedder": "HashingEmbedder"})


def test_remove_source_and_hash_reuse():
    store = _store_with(["alpha", "beta", "gamma"])
    assert store.known_hashes() == {"h0", "h1", "h2"}
    reused = store.vectors_for_hashes({"h1"})
    assert set(reused) == {"h1"}
    removed = store.remove_source("doc1.md")
    assert removed == 1
    assert len(store) == 2
    assert store.known_hashes() == {"h0", "h2"}
    # search still works and never returns the removed record
    hits = store.search(HashingEmbedder().embed(["beta"])[0], k=2)
    assert all(store.records[i]["source"] != "doc1.md" for i, _ in hits)


def test_dimension_mismatch_is_rejected():
    store = _store_with(["alpha"])
    with pytest.raises(ValueError):
        store.add(np.zeros((1, 8), dtype=np.float32), [{"text": "x", "source": "s"}])
