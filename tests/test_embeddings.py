"""Tests for the embeddings module."""

import pytest
from src.embeddings import (
    generate_embedding,
    generate_embeddings_batch,
    cosine_similarity,
    EMBEDDING_DIMENSION,
)
from cost_tracker import CostTracker


class TestGenerateEmbedding:
    """Tests for single embedding generation (calls OpenAI API)."""

    def test_returns_correct_dimensions(self):
        """Embedding should have 1536 dimensions."""
        embedding = generate_embedding("Hello world")
        assert len(embedding) == EMBEDDING_DIMENSION

    def test_returns_floats(self):
        """Each element should be a float."""
        embedding = generate_embedding("Test text")
        assert all(isinstance(x, float) for x in embedding)

    def test_with_cost_tracker(self):
        """Should track costs when a tracker is provided."""
        tracker = CostTracker(log_file="test_costs.json")
        embedding = generate_embedding("Track this cost", tracker=tracker)
        assert len(embedding) == EMBEDDING_DIMENSION
        assert len(tracker.session_costs) == 1
        assert tracker.session_costs[0]["model"] == "embedding-small"


class TestGenerateEmbeddingsBatch:
    """Tests for batch embedding generation."""

    def test_batch_returns_correct_count(self):
        """Should return one embedding per input text."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = generate_embeddings_batch(texts)
        assert len(embeddings) == 3

    def test_batch_correct_dimensions(self):
        """Each embedding in the batch should have 1536 dimensions."""
        texts = ["Hello", "World"]
        embeddings = generate_embeddings_batch(texts)
        for emb in embeddings:
            assert len(emb) == EMBEDDING_DIMENSION

    def test_empty_batch(self):
        """Empty input should return empty list without API call."""
        embeddings = generate_embeddings_batch([])
        assert embeddings == []


class TestCosineSimilarity:
    """Tests for cosine similarity (no API calls needed)."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Perpendicular vectors should have similarity of 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_similar_texts_have_higher_similarity(self):
        """Semantically similar texts should score higher than unrelated ones."""
        emb_ai = generate_embedding("artificial intelligence")
        emb_ml = generate_embedding("machine learning")
        emb_pizza = generate_embedding("pepperoni pizza recipe")

        sim_related = cosine_similarity(emb_ai, emb_ml)
        sim_unrelated = cosine_similarity(emb_ai, emb_pizza)

        assert sim_related > sim_unrelated, (
            f"Related ({sim_related:.4f}) should be > unrelated ({sim_unrelated:.4f})"
        )

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0
