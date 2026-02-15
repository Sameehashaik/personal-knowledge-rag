# test_vector_store.py — add, search, save/load with small 4-dim vectors (no API)

import os
import pytest
import numpy as np
from src.vector_store import VectorStore


def _make_random_embeddings(n: int, dim: int = 1536) -> list[list[float]]:
    """Random vectors for tests that don't care about meaning."""
    return np.random.rand(n, dim).astype(np.float32).tolist()


class TestAddDocuments:

    def test_add_single_document(self):
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello world"],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
        )
        assert store.count == 1

    def test_add_multiple_documents(self):
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Doc 1", "Doc 2", "Doc 3"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )
        assert store.count == 3

    def test_add_with_metadata(self):
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello"],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
            metadata=[{"source": "test.txt", "chunk_index": 0}],
        )
        assert store.metadata[0]["source"] == "test.txt"

    def test_mismatched_lengths_raises(self):
        store = VectorStore(dimension=4)
        with pytest.raises(ValueError, match="Mismatch"):
            store.add_documents(
                texts=["One", "Two"],
                embeddings=[[1, 0, 0, 0]],
            )


class TestSearch:

    def test_search_returns_results(self):
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Doc A", "Doc B", "Doc C"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )
        results = store.search([1, 0, 0, 0], k=2)
        assert len(results) == 2

    def test_search_finds_most_similar(self):
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Exact match", "Somewhat close", "Very different"],
            embeddings=[[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 0, 1]],
        )
        results = store.search([1, 0, 0, 0], k=3)
        assert results[0]["text"] == "Exact match"
        assert results[0]["rank"] == 1

    def test_search_returns_distance(self):
        # exact match should have distance ~0
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Test"],
            embeddings=[[1, 0, 0, 0]],
        )
        results = store.search([1, 0, 0, 0], k=1)
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_search_empty_store(self):
        store = VectorStore(dimension=4)
        results = store.search([1, 0, 0, 0], k=3)
        assert results == []

    def test_k_larger_than_store(self):
        # asking for k=10 when there's only 1 doc should just return 1
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Only one"],
            embeddings=[[1, 0, 0, 0]],
        )
        results = store.search([1, 0, 0, 0], k=10)
        assert len(results) == 1


class TestSaveLoad:

    def test_save_and_load(self, tmp_path):
        filepath = str(tmp_path / "test_store")

        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello", "World"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadata=[{"id": 1}, {"id": 2}],
        )
        store.save(filepath)

        loaded = VectorStore.load(filepath)
        assert loaded.count == 2
        assert loaded.texts == ["Hello", "World"]
        assert loaded.metadata[0]["id"] == 1

    def test_loaded_store_searchable(self, tmp_path):
        # make sure the FAISS index actually works after round-tripping to disk
        filepath = str(tmp_path / "test_store")

        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Target doc", "Other doc"],
            embeddings=[[1, 0, 0, 0], [0, 0, 0, 1]],
        )
        store.save(filepath)

        loaded = VectorStore.load(filepath)
        results = loaded.search([1, 0, 0, 0], k=1)
        assert results[0]["text"] == "Target doc"
