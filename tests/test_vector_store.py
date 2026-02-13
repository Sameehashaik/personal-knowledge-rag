"""Tests for the vector store module."""

import os
import pytest
import numpy as np
from src.vector_store import VectorStore


def _make_random_embeddings(n: int, dim: int = 1536) -> list[list[float]]:
    """Helper: generate random embeddings for testing (no API call needed)."""
    return np.random.rand(n, dim).astype(np.float32).tolist()


class TestAddDocuments:
    """Tests for adding documents to the store."""

    def test_add_single_document(self):
        """Should store one document."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello world"],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
        )
        assert store.count == 1

    def test_add_multiple_documents(self):
        """Should store multiple documents."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Doc 1", "Doc 2", "Doc 3"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )
        assert store.count == 3

    def test_add_with_metadata(self):
        """Should store metadata alongside documents."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello"],
            embeddings=[[1.0, 2.0, 3.0, 4.0]],
            metadata=[{"source": "test.txt", "chunk_index": 0}],
        )
        assert store.metadata[0]["source"] == "test.txt"

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError if texts and embeddings don't match."""
        store = VectorStore(dimension=4)
        with pytest.raises(ValueError, match="Mismatch"):
            store.add_documents(
                texts=["One", "Two"],
                embeddings=[[1, 0, 0, 0]],
            )


class TestSearch:
    """Tests for similarity search."""

    def test_search_returns_results(self):
        """Should return search results."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Doc A", "Doc B", "Doc C"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )
        results = store.search([1, 0, 0, 0], k=2)
        assert len(results) == 2

    def test_search_finds_most_similar(self):
        """The closest vector should be ranked first."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Exact match", "Somewhat close", "Very different"],
            embeddings=[[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 0, 1]],
        )
        results = store.search([1, 0, 0, 0], k=3)
        assert results[0]["text"] == "Exact match"
        assert results[0]["rank"] == 1

    def test_search_returns_distance(self):
        """Results should include a distance score."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Test"],
            embeddings=[[1, 0, 0, 0]],
        )
        results = store.search([1, 0, 0, 0], k=1)
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_search_empty_store(self):
        """Searching an empty store should return empty list."""
        store = VectorStore(dimension=4)
        results = store.search([1, 0, 0, 0], k=3)
        assert results == []

    def test_k_larger_than_store(self):
        """Requesting more results than documents should not crash."""
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Only one"],
            embeddings=[[1, 0, 0, 0]],
        )
        results = store.search([1, 0, 0, 0], k=10)
        assert len(results) == 1


class TestSaveLoad:
    """Tests for saving and loading the store."""

    def test_save_and_load(self, tmp_path):
        """Should be able to save and reload the store."""
        filepath = str(tmp_path / "test_store")

        # Create and save
        store = VectorStore(dimension=4)
        store.add_documents(
            texts=["Hello", "World"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadata=[{"id": 1}, {"id": 2}],
        )
        store.save(filepath)

        # Load into a new instance
        loaded = VectorStore.load(filepath)
        assert loaded.count == 2
        assert loaded.texts == ["Hello", "World"]
        assert loaded.metadata[0]["id"] == 1

    def test_loaded_store_searchable(self, tmp_path):
        """A loaded store should still be searchable."""
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
