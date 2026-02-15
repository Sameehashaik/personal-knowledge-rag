# test_embeddings.py — single embed, batch embed, cosine math
# NOTE: tests that call generate_embedding() hit the real OpenAI API (costs money)

import pytest
from src.embeddings import (
    generate_embedding,
    generate_embeddings_batch,
    cosine_similarity,
    EMBEDDING_DIMENSION,
)
from cost_tracker import CostTracker


class TestGenerateEmbedding:
    # these hit the OpenAI API — they'll fail without a valid key

    def test_returns_correct_dimensions(self):
        embedding = generate_embedding("Hello world")
        assert len(embedding) == EMBEDDING_DIMENSION

    def test_returns_floats(self):
        embedding = generate_embedding("Test text")
        assert all(isinstance(x, float) for x in embedding)

    def test_with_cost_tracker(self):
        tracker = CostTracker(log_file="test_costs.json")
        embedding = generate_embedding("Track this cost", tracker=tracker)
        assert len(embedding) == EMBEDDING_DIMENSION
        assert len(tracker.session_costs) == 1
        assert tracker.session_costs[0]["model"] == "embedding-small"


class TestGenerateEmbeddingsBatch:

    def test_batch_returns_correct_count(self):
        texts = ["First text", "Second text", "Third text"]
        embeddings = generate_embeddings_batch(texts)
        assert len(embeddings) == 3

    def test_batch_correct_dimensions(self):
        texts = ["Hello", "World"]
        embeddings = generate_embeddings_batch(texts)
        for emb in embeddings:
            assert len(emb) == EMBEDDING_DIMENSION

    def test_empty_batch(self):
        # empty list should short-circuit without calling the API
        embeddings = generate_embeddings_batch([])
        assert embeddings == []


class TestCosineSimilarity:
    # pure math — no API calls here

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_similar_texts_have_higher_similarity(self):
        # "AI" and "ML" should be closer than "AI" and "pizza"
        emb_ai = generate_embedding("artificial intelligence")
        emb_ml = generate_embedding("machine learning")
        emb_pizza = generate_embedding("pepperoni pizza recipe")

        sim_related = cosine_similarity(emb_ai, emb_ml)
        sim_unrelated = cosine_similarity(emb_ai, emb_pizza)

        assert sim_related > sim_unrelated, (
            f"Related ({sim_related:.4f}) should be > unrelated ({sim_unrelated:.4f})"
        )

    def test_zero_vector(self):
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0
