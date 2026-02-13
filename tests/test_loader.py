"""Tests for the document loader module."""

import pytest
from pathlib import Path
from src.document_loader import load_txt, load_document, clean_text

# Path to our sample data
SAMPLE_TXT = str(Path(__file__).parent.parent / "data" / "sample1.txt")


class TestLoadTxt:
    """Tests for the load_txt function."""

    def test_load_sample_file(self):
        """Should successfully load our sample text file."""
        text = load_txt(SAMPLE_TXT)
        assert len(text) > 0
        assert "RAG" in text

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_txt("nonexistent_file.txt")

    def test_wrong_extension(self):
        """Should raise ValueError for non-.txt files."""
        with pytest.raises(ValueError):
            load_txt("document.pdf")


class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_extra_newlines(self):
        """Multiple blank lines should be collapsed to one."""
        dirty = "Hello\n\n\n\n\nWorld"
        cleaned = clean_text(dirty)
        assert cleaned == "Hello\n\nWorld"

    def test_removes_extra_spaces(self):
        """Multiple spaces should be collapsed to one."""
        dirty = "Hello     World"
        cleaned = clean_text(dirty)
        assert cleaned == "Hello World"

    def test_strips_whitespace(self):
        """Leading and trailing whitespace should be removed."""
        dirty = "   Hello World   "
        cleaned = clean_text(dirty)
        assert cleaned == "Hello World"

    def test_strips_line_whitespace(self):
        """Leading/trailing whitespace on each line should be removed."""
        dirty = "  Line 1  \n  Line 2  "
        cleaned = clean_text(dirty)
        assert cleaned == "Line 1\nLine 2"


class TestLoadDocument:
    """Tests for the load_document auto-detect function."""

    def test_loads_txt_file(self):
        """Should auto-detect .txt and load it."""
        text = load_document(SAMPLE_TXT)
        assert "Retrieval-Augmented Generation" in text

    def test_unsupported_type(self):
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("notes.docx")

    def test_returned_text_is_cleaned(self):
        """load_document should return cleaned text."""
        text = load_document(SAMPLE_TXT)
        # Should not have triple+ newlines (clean_text removes them)
        assert "\n\n\n" not in text
