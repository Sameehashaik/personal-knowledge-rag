"""Tests for the RAG chain module (end-to-end tests, calls OpenAI API)."""

import pytest
from pathlib import Path
from src.rag_chain import RAGChain

SAMPLE_TXT = str(Path(__file__).parent.parent / "data" / "sample1.txt")


class TestAddDocument:
    """Tests for document ingestion."""

    def test_add_txt_document(self):
        """Should ingest a text file and return a summary."""
        rag = RAGChain()
        summary = rag.add_document(SAMPLE_TXT)

        assert summary["filename"] == "sample1.txt"
        assert summary["chunks"] > 0
        assert summary["characters"] > 0
        assert summary["total_chunks_in_store"] == summary["chunks"]

    def test_add_tracks_filename(self):
        """Should track which documents have been added."""
        rag = RAGChain()
        rag.add_document(SAMPLE_TXT)
        assert "sample1.txt" in rag.documents


class TestQuery:
    """Tests for question answering."""

    @pytest.fixture
    def loaded_rag(self):
        """A RAG chain with sample1.txt already loaded."""
        rag = RAGChain()
        rag.add_document(SAMPLE_TXT)
        return rag

    def test_query_returns_answer(self, loaded_rag):
        """Should return a non-empty answer."""
        result = loaded_rag.query("What is RAG?")
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, loaded_rag):
        """Should return source citations."""
        result = loaded_rag.query("What is RAG?")
        assert len(result["sources"]) > 0
        assert result["sources"][0]["source"] == "sample1.txt"

    def test_query_returns_cost(self, loaded_rag):
        """Should track and return the cost."""
        result = loaded_rag.query("What is RAG?")
        assert result["cost"] is not None
        assert result["cost"] >= 0

    def test_query_empty_store(self):
        """Should handle questions when no documents are loaded."""
        rag = RAGChain()
        result = rag.query("What is anything?")
        assert "No documents" in result["answer"]
        assert result["chunks_used"] == 0

    def test_out_of_scope_question(self, loaded_rag):
        """Question not in document should say it can't answer."""
        result = loaded_rag.query("What is the capital of Mars?")
        answer_lower = result["answer"].lower()
        # The LLM should indicate it can't find the answer
        assert any(phrase in answer_lower for phrase in [
            "don't have enough information",
            "not mentioned",
            "no information",
            "doesn't contain",
            "does not contain",
            "not covered",
            "cannot answer",
            "can't answer",
            "not in the provided",
        ]), f"Expected refusal, got: {result['answer'][:200]}"


class TestBuildPrompt:
    """Tests for prompt construction (no API calls)."""

    def test_prompt_includes_question(self):
        """The prompt should contain the user's question."""
        rag = RAGChain()
        chunks = [{"text": "Some context.", "metadata": {"source": "test.txt"}}]
        prompt = rag.build_rag_prompt("What is X?", chunks)
        assert "What is X?" in prompt

    def test_prompt_includes_context(self):
        """The prompt should contain the retrieved chunks."""
        rag = RAGChain()
        chunks = [{"text": "RAG is great.", "metadata": {"source": "doc.txt"}}]
        prompt = rag.build_rag_prompt("Tell me about RAG", chunks)
        assert "RAG is great." in prompt

    def test_prompt_includes_source_labels(self):
        """Each chunk should be labeled with its source."""
        rag = RAGChain()
        chunks = [
            {"text": "Chunk 1", "metadata": {"source": "a.txt"}},
            {"text": "Chunk 2", "metadata": {"source": "b.txt"}},
        ]
        prompt = rag.build_rag_prompt("Question?", chunks)
        assert "[Source 1: a.txt]" in prompt
        assert "[Source 2: b.txt]" in prompt
