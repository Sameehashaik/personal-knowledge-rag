"""
Embeddings - Convert text into numerical vectors using OpenAI's API.

This is the "magic translation" step: we turn human-readable text into
lists of numbers (vectors) that capture the *meaning* of the text.
Similar meanings → similar numbers → we can find relevant chunks by math.
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add the project root to find cost_tracker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker

# Load API key from .env file
load_dotenv()

# Model we'll use: small, cheap, and good enough for RAG
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def _get_client() -> OpenAI:
    """Create an OpenAI client using the API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Make sure it's set in your .env file."
        )
    return OpenAI(api_key=api_key)


def generate_embedding(text: str, tracker: CostTracker = None) -> list[float]:
    """
    Generate an embedding vector for a single piece of text.

    Args:
        text: The text to embed.
        tracker: Optional CostTracker to log the API cost.

    Returns:
        A list of 1536 floats representing the text's meaning.
    """
    client = _get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    embedding = response.data[0].embedding

    # Track the cost if a tracker was provided
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
    Generate embeddings for multiple texts in a single API call.

    This is more efficient than calling generate_embedding() in a loop
    because the API can process them all at once.

    Args:
        texts: List of texts to embed.
        tracker: Optional CostTracker to log the API cost.

    Returns:
        A list of embedding vectors (one per input text).
    """
    if not texts:
        return []

    client = _get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    # Track the cost
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
    Calculate cosine similarity between two vectors.

    Returns a value between -1 and 1:
      1.0 = identical meaning
      0.0 = completely unrelated
     -1.0 = opposite meaning

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Demo when running directly
if __name__ == "__main__":
    tracker = CostTracker(log_file="project1_costs.json")

    print("Generating embeddings for 3 phrases...\n")

    words = ["artificial intelligence", "AI and machine learning", "pizza recipe"]
    embeddings = generate_embeddings_batch(words, tracker=tracker)

    print(f"Each embedding has {len(embeddings[0])} dimensions\n")

    # Show similarity scores
    print("Similarity scores:")
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            print(f'  "{words[i]}" vs "{words[j]}": {score:.4f}')

    print()
    tracker.print_session_summary()
