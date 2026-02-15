# document_loader.py
# Step 1 of the RAG pipeline: get raw text out of files (.txt, .pdf) and clean it up.
# Everything downstream depends on this producing clean, readable text.

from pathlib import Path
from PyPDF2 import PdfReader
import re


def load_txt(file_path: str) -> str:
    """Read a .txt file and return its raw content. Validates extension + existence first."""
    path = Path(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {path.suffix}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # utf-8 so accented chars / special symbols don't break
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Extract text from every page of a PDF using PyPDF2. Skips blank pages."""
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
    Normalize whitespace so the chunker gets consistent input.
    Collapses triple+ newlines, double+ spaces, and strips each line.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    text = re.sub(r" {2,}", " ", text)        # collapse runs of spaces
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = text.strip()

    return text


def load_document(file_path: str) -> str:
    """
    Main entry point — detects .txt vs .pdf, loads it, and returns cleaned text.
    Raises ValueError for anything other than .txt / .pdf.
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


# --- quick manual test: python document_loader.py <file> ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <file_path>")
        sys.exit(1)

    text = load_document(sys.argv[1])
    print(f"Loaded {len(text)} characters")
    print(f"First 200 chars:\n{text[:200]}...")
