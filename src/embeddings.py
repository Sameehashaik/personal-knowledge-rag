# embeddings.py
# Turns text into 1536-dim vectors via OpenAI's embedding API.
# Similar text -> nearby vectors -> that's how retrieval finds relevant chunks.

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# need project root on path so we can import cost_tracker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker

load_dotenv()

# text-embedding-3-small: cheap, fast, 1536 dims — good enough for this use case
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def _get_client() -> OpenAI:
    """Build an OpenAI client from the OPENAI_API_KEY env var."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Make sure it's set in your .env file."
        )
    return OpenAI(api_key=api_key)


def generate_embedding(text: str, tracker: CostTracker = None) -> list[float]:
    """
    Embed a single string. Returns a 1536-float vector.
    Pass a CostTracker to log the API spend.
    """
    client = _get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    embedding = response.data[0].embedding

    if tracker:
        tokens_used = response.usage.total_tokens
        tracker.track_call(
            model="embedding-small",
            input_tokens=tokens_used,
            output_tokens=0,
            description=f"Embed single text ({len(text)} chars)",
        )

    return embedding


def generate_embeddings_batch(
    texts: list[str], tracker: CostTracker = None
) -> list[list[float]]:
    """
    Embed multiple texts in one API call — way cheaper than looping generate_embedding().
    Returns one vector per input text in the same order.
    """
    if not texts:
        return []

    client = _get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    if tracker:
        tokens_used = response.usage.total_tokens
        tracker.track_call(
            model="embedding-small",
            input_tokens=tokens_used,
            output_tokens=0,
            description=f"Embed batch of {len(texts)} texts",
        )

    return embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    1.0 = same direction (same meaning), 0.0 = unrelated, -1.0 = opposite.
    Returns 0.0 if either vector is all zeros.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# --- quick demo: embed 3 phrases and compare them ---
if __name__ == "__main__":
    tracker = CostTracker(log_file="project1_costs.json")

    print("Generating embeddings for 3 phrases...\n")

    words = ["artificial intelligence", "AI and machine learning", "pizza recipe"]
    embeddings = generate_embeddings_batch(words, tracker=tracker)

    print(f"Each embedding has {len(embeddings[0])} dimensions\n")

    print("Similarity scores:")
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            print(f'  "{words[i]}" vs "{words[j]}": {score:.4f}')

    print()
    tracker.print_session_summary()
