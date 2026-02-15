# test_loader.py — make sure document_loader handles txt/pdf, bad paths, and cleaning

import pytest
from pathlib import Path
from src.document_loader import load_txt, load_document, clean_text

SAMPLE_TXT = str(Path(__file__).parent.parent / "data" / "sample1.txt")


class TestLoadTxt:

    def test_load_sample_file(self):
        text = load_txt(SAMPLE_TXT)
        assert len(text) > 0
        assert "RAG" in text

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_txt("nonexistent_file.txt")

    def test_wrong_extension(self):
        # passing a .pdf path to load_txt should blow up
        with pytest.raises(ValueError):
            load_txt("document.pdf")


class TestCleanText:

    def test_removes_extra_newlines(self):
        dirty = "Hello\n\n\n\n\nWorld"
        cleaned = clean_text(dirty)
        assert cleaned == "Hello\n\nWorld"

    def test_removes_extra_spaces(self):
        dirty = "Hello     World"
        cleaned = clean_text(dirty)
        assert cleaned == "Hello World"

    def test_strips_whitespace(self):
        dirty = "   Hello World   "
        cleaned = clean_text(dirty)
        assert cleaned == "Hello World"

    def test_strips_line_whitespace(self):
        dirty = "  Line 1  \n  Line 2  "
        cleaned = clean_text(dirty)
        assert cleaned == "Line 1\nLine 2"


class TestLoadDocument:

    def test_loads_txt_file(self):
        text = load_document(SAMPLE_TXT)
        assert "Retrieval-Augmented Generation" in text

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("notes.docx")

    def test_returned_text_is_cleaned(self):
        # clean_text collapses triple+ newlines, so none should remain
        text = load_document(SAMPLE_TXT)
        assert "\n\n\n" not in text
