# test_chunker.py — verify splitting, overlap, edge cases, and metadata generation

import pytest
from src.chunker import chunk_text, get_chunk_metadata


class TestChunkText:

    def test_basic_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=35, overlap=10)
        assert len(chunks) >= 2

    def test_all_text_preserved(self):
        # every keyword should show up in at least one chunk
        text = "Alpha fact. Bravo fact. Charlie fact. Delta fact. Echo fact."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        combined = " ".join(chunks)
        for word in ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]:
            assert word in combined, f"'{word}' missing from chunks"

    def test_overlap_exists(self):
        # tail of chunk N should appear at the head of chunk N+1
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
            "The five boxing wizards jump quickly."
        )
        chunks = chunk_text(text, chunk_size=60, overlap=20)
        if len(chunks) >= 2:
            overlap_found = False
            words_end = chunks[0].split()[-3:]
            for word in words_end:
                if word in chunks[1]:
                    overlap_found = True
                    break
            assert overlap_found, "Expected overlap between consecutive chunks"

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_small_text_single_chunk(self):
        text = "Short text here."
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == "Short text here."

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=0)

    def test_overlap_larger_than_chunk(self):
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=100, overlap=100)

    def test_negative_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("Hello.", chunk_size=100, overlap=-1)


class TestGetChunkMetadata:

    def test_metadata_count_matches_chunks(self):
        chunks = ["Chunk one text.", "Chunk two text.", "Chunk three text."]
        meta = get_chunk_metadata(chunks)
        assert len(meta) == 3

    def test_metadata_fields(self):
        chunks = ["Hello world, this is a test."]
        meta = get_chunk_metadata(chunks)
        assert meta[0]["chunk_index"] == 0
        assert meta[0]["char_count"] == len(chunks[0])
        assert meta[0]["word_count"] == 6

    def test_empty_chunks(self):
        assert get_chunk_metadata([]) == []
