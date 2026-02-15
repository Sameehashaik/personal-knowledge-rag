# test_rag_chain.py — end-to-end: ingest, query, prompt building
# WARNING: most of these hit the OpenAI API (embedding + chat completion)

import pytest
from pathlib import Path
from src.rag_chain import RAGChain

SAMPLE_TXT = str(Path(__file__).parent.parent / "data" / "sample1.txt")


class TestAddDocument:

    def test_add_txt_document(self):
        rag = RAGChain()
        summary = rag.add_document(SAMPLE_TXT)

        assert summary["filename"] == "sample1.txt"
        assert summary["chunks"] > 0
        assert summary["characters"] > 0
        assert summary["total_chunks_in_store"] == summary["chunks"]

    def test_add_tracks_filename(self):
        rag = RAGChain()
        rag.add_document(SAMPLE_TXT)
        assert "sample1.txt" in rag.documents


class TestQuery:

    @pytest.fixture
    def loaded_rag(self):
        """Pre-load sample1.txt so every test in this class can query it."""
        rag = RAGChain()
        rag.add_document(SAMPLE_TXT)
        return rag

    def test_query_returns_answer(self, loaded_rag):
        result = loaded_rag.query("What is RAG?")
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, loaded_rag):
        result = loaded_rag.query("What is RAG?")
        assert len(result["sources"]) > 0
        assert result["sources"][0]["source"] == "sample1.txt"

    def test_query_returns_cost(self, loaded_rag):
        result = loaded_rag.query("What is RAG?")
        assert result["cost"] is not None
        assert result["cost"] >= 0

    def test_query_empty_store(self):
        # no docs loaded — should get a polite "nothing here" message
        rag = RAGChain()
        result = rag.query("What is anything?")
        assert "No documents" in result["answer"]
        assert result["chunks_used"] == 0

    def test_out_of_scope_question(self, loaded_rag):
        # question has nothing to do with the loaded doc — LLM should refuse
        result = loaded_rag.query("What is the capital of Mars?")
        answer_lower = result["answer"].lower()
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
    # no API calls — just string assembly

    def test_prompt_includes_question(self):
        rag = RAGChain()
        chunks = [{"text": "Some context.", "metadata": {"source": "test.txt"}}]
        prompt = rag.build_rag_prompt("What is X?", chunks)
        assert "What is X?" in prompt

    def test_prompt_includes_context(self):
        rag = RAGChain()
        chunks = [{"text": "RAG is great.", "metadata": {"source": "doc.txt"}}]
        prompt = rag.build_rag_prompt("Tell me about RAG", chunks)
        assert "RAG is great." in prompt

    def test_prompt_includes_source_labels(self):
        rag = RAGChain()
        chunks = [
            {"text": "Chunk 1", "metadata": {"source": "a.txt"}},
            {"text": "Chunk 2", "metadata": {"source": "b.txt"}},
        ]
        prompt = rag.build_rag_prompt("Question?", chunks)
        assert "[Source 1: a.txt]" in prompt
        assert "[Source 2: b.txt]" in prompt
