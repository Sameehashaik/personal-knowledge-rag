"""
Text Chunker - Split documents into overlapping chunks for RAG.

Why chunk? LLMs have a limited context window. We can't send a 50-page document
all at once. Instead, we split it into small pieces and only retrieve the
*relevant* pieces for each question.
"""

import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks, breaking on sentence boundaries.

    Args:
        text: The full text to split.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split text into sentences using regex
    # This pattern matches sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed chunk_size and we already
        # have content, finalize the current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build the overlap: walk backwards through sentences until
            # we've accumulated enough characters for the overlap.
            # Always include at least the last sentence so there's
            # some continuity between chunks.
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_chunk_metadata(chunks: list[str]) -> list[dict]:
    """
    Generate metadata for each chunk (useful for tracking and debugging).

    Args:
        chunks: List of text chunks.

    Returns:
        List of dicts with metadata for each chunk.
    """
    metadata = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        metadata.append({
            "chunk_index": i,
            "char_count": len(chunk),
            "word_count": len(words),
            "first_words": " ".join(words[:8]) + "..." if len(words) > 8 else chunk,
        })
    return metadata


# Quick demo when running directly
if __name__ == "__main__":
    from pathlib import Path

    # Load our sample document
    sample_path = Path(__file__).parent.parent / "data" / "sample1.txt"
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Original text: {len(text)} characters\n")

    # Demo with different chunk sizes
    for size in [500, 1000, 2000]:
        chunks = chunk_text(text, chunk_size=size, overlap=200)
        print(f"Chunk size={size}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk)} chars")
    print()

    # Show the first chunk in detail
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    meta = get_chunk_metadata(chunks)
    print("--- Detailed view (chunk_size=1000) ---")
    for m in meta:
        print(f"  Chunk {m['chunk_index']}: {m['char_count']} chars, "
              f"{m['word_count']} words | {m['first_words']}")
