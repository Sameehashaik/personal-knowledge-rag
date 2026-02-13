"""
Document Loader - Load and clean text from TXT and PDF files.
This is the first step in the RAG pipeline: getting text INTO the system.
"""

from pathlib import Path
from PyPDF2 import PdfReader
import re


def load_txt(file_path: str) -> str:
    """
    Load text content from a .txt file.

    Args:
        file_path: Path to the text file.

    Returns:
        The raw text content of the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a .txt file.
    """
    path = Path(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {path.suffix}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # encoding='utf-8' ensures we handle special characters properly
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """
    Load text content from a PDF file using PyPDF2.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The extracted text from all pages.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a .pdf file.
    """
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and artifacts.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text.
    """
    # Replace multiple newlines with a single one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with a single space
    text = re.sub(r" {2,}", " ", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace from the whole text
    text = text.strip()

    return text


def load_document(file_path: str) -> str:
    """
    Auto-detect file type and load the document.

    This is the main entry point - it figures out whether you gave it
    a TXT or PDF file and calls the right loader.

    Args:
        file_path: Path to a .txt or .pdf file.

    Returns:
        Cleaned text content of the document.

    Raises:
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        raw_text = load_txt(file_path)
    elif suffix == ".pdf":
        raw_text = load_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: .txt, .pdf"
        )

    return clean_text(raw_text)


# Quick test when running this file directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <file_path>")
        sys.exit(1)

    text = load_document(sys.argv[1])
    print(f"Loaded {len(text)} characters")
    print(f"First 200 chars:\n{text[:200]}...")
