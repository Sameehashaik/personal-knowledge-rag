"""Tests for the text chunker module."""

import pytest
from src.chunker import chunk_text, get_chunk_metadata


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_basic_chunking(self):
        """Should split text into multiple chunks."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=35, overlap=10)
        assert len(chunks) >= 2

    def test_all_text_preserved(self):
        """Every sentence from the original text should appear in some chunk."""
        text = "Alpha fact. Bravo fact. Charlie fact. Delta fact. Echo fact."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        combined = " ".join(chunks)
        for word in ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]:
            assert word in combined, f"'{word}' missing from chunks"

    def test_overlap_exists(self):
        """Consecutive chunks should share some overlapping text."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
            "The five boxing wizards jump quickly."
        )
        chunks = chunk_text(text, chunk_size=60, overlap=20)
        if len(chunks) >= 2:
            # Check that the end of chunk 0 overlaps with the start of chunk 1
            # by finding shared content
            overlap_found = False
            words_end = chunks[0].split()[-3:]  # last 3 words of chunk 0
            for word in words_end:
                if word in chunks[1]:
                    overlap_found = True
                    break
            assert overlap_found, "Expected overlap between consecutive chunks"

    def test_empty_text(self):
        """Empty or whitespace-only text should return no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_small_text_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        text = "Short text here."
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == "Short text here."

    def test_invalid_chunk_size(self):
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=0)

    def test_overlap_larger_than_chunk(self):
        """Should raise ValueError if overlap >= chunk_size."""
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=100, overlap=100)

    def test_negative_overlap(self):
        """Should raise ValueError for negative overlap."""
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=100, overlap=-1)


class TestGetChunkMetadata:
    """Tests for the get_chunk_metadata function."""

    def test_metadata_count_matches_chunks(self):
        """Should return one metadata dict per chunk."""
        chunks = ["Chunk one text.", "Chunk two text.", "Chunk three text."]
        meta = get_chunk_metadata(chunks)
        assert len(meta) == 3

    def test_metadata_fields(self):
        """Each metadata dict should have the expected fields."""
        chunks = ["Hello world, this is a test."]
        meta = get_chunk_metadata(chunks)
        assert meta[0]["chunk_index"] == 0
        assert meta[0]["char_count"] == len(chunks[0])
        assert meta[0]["word_count"] == 6

    def test_empty_chunks(self):
        """Should handle empty list."""
        assert get_chunk_metadata([]) == []
