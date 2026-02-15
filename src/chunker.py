# chunker.py
# Step 2 of RAG pipeline: break long text into smaller overlapping pieces.
# Overlap lets the retriever catch context that sits on a chunk boundary.

import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks on sentence boundaries.
    chunk_size and overlap are in characters. Returns [] for blank input.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # chunk is full — seal it and carry over sentences for overlap
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # walk backwards through sentences to build the overlap window;
            # always keep at least one sentence so chunks aren't totally disjoint
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

    # last chunk (whatever's left)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_chunk_metadata(chunks: list[str]) -> list[dict]:
    """Return index/char/word counts and a preview for each chunk — useful for debugging."""
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


# --- quick demo: run this file directly to see chunking in action ---
if __name__ == "__main__":
    from pathlib import Path

    sample_path = Path(__file__).parent.parent / "data" / "sample1.txt"
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Original text: {len(text)} characters\n")

    for size in [500, 1000, 2000]:
        chunks = chunk_text(text, chunk_size=size, overlap=200)
        print(f"Chunk size={size}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk)} chars")
    print()

    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    meta = get_chunk_metadata(chunks)
    print("--- Detailed view (chunk_size=1000) ---")
    for m in meta:
        print(f"  Chunk {m['chunk_index']}: {m['char_count']} chars, "
              f"{m['word_count']} words | {m['first_words']}")
