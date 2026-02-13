"""
Vector Store - Store and search embeddings using FAISS.

This is the "database" of our RAG system. Instead of searching by keywords
(like Google), it searches by *meaning*. You give it a question embedding,
and it finds the chunks whose embeddings are closest in 1536-dimensional space.
"""

import json
import numpy as np
import faiss
from pathlib import Path


class VectorStore:
    """
    A vector database backed by FAISS for fast similarity search.

    Stores three things together:
    1. The embedding vectors (in FAISS index)
    2. The original text chunks (so we can return readable results)
    3. Metadata about each chunk (source file, chunk index, etc.)
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize an empty vector store.

        Args:
            dimension: Size of each embedding vector (1536 for OpenAI small).
        """
        self.dimension = dimension
        # IndexFlatL2 = brute-force L2 (Euclidean) distance search
        # Simple and exact — perfect for our scale (hundreds/thousands of chunks)
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict] = None,
    ) -> None:
        """
        Add documents (chunks) to the vector store.

        Args:
            texts: The original text chunks.
            embeddings: The embedding vectors for each chunk.
            metadata: Optional metadata dicts for each chunk.
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(embeddings)} embeddings"
            )

        if metadata and len(metadata) != len(texts):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(metadata)} metadata entries"
            )

        # Convert to numpy array — FAISS requires float32
        vectors = np.array(embeddings, dtype=np.float32)

        # Add vectors to the FAISS index
        self.index.add(vectors)

        # Store the texts and metadata alongside
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

    def search(self, query_embedding: list[float], k: int = 3) -> list[dict]:
        """
        Find the k most similar chunks to a query.

        Args:
            query_embedding: The embedding vector of the query/question.
            k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata, distance, rank.
            Results are sorted by relevance (lowest distance = most similar).
        """
        if self.index.ntotal == 0:
            return []

        # Don't request more results than we have documents
        k = min(k, self.index.ntotal)

        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)

        # FAISS search returns distances and indices
        distances, indices = self.index.search(query_vector, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "distance": float(dist),
                "rank": rank + 1,
            })

        return results

    @property
    def count(self) -> int:
        """Number of documents in the store."""
        return self.index.ntotal

    def save(self, filepath: str) -> None:
        """
        Save the vector store to disk (two files: .faiss + .json).

        Args:
            filepath: Base path (without extension). Creates:
                      filepath.faiss — the FAISS index
                      filepath.json  — the texts and metadata
        """
        path = Path(filepath)

        # Save the FAISS index
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        # Save texts and metadata as JSON
        store_data = {
            "dimension": self.dimension,
            "texts": self.texts,
            "metadata": self.metadata,
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(store_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """
        Load a vector store from disk.

        Args:
            filepath: Base path (without extension).

        Returns:
            A VectorStore with the saved data restored.
        """
        path = Path(filepath)

        # Load texts and metadata
        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            store_data = json.load(f)

        # Create a new VectorStore and restore its state
        store = cls(dimension=store_data["dimension"])
        store.texts = store_data["texts"]
        store.metadata = store_data["metadata"]

        # Load the FAISS index
        store.index = faiss.read_index(str(path.with_suffix(".faiss")))

        return store


# Demo when running directly
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.embeddings import generate_embeddings_batch, generate_embedding, cosine_similarity
    from cost_tracker import CostTracker

    tracker = CostTracker(log_file="project1_costs.json")

    # Create some sample chunks
    chunks = [
        "RAG stands for Retrieval-Augmented Generation. It combines LLMs with external knowledge.",
        "Embeddings are numerical representations of text. Similar meanings produce similar vectors.",
        "FAISS is a library by Meta for efficient similarity search in high-dimensional spaces.",
        "Python is a popular programming language used in data science and machine learning.",
        "Pizza is made with dough, tomato sauce, and cheese, then baked in an oven.",
    ]

    print("Adding 5 chunks to vector store...\n")
    embeddings = generate_embeddings_batch(chunks, tracker=tracker)

    store = VectorStore(dimension=1536)
    store.add_documents(
        texts=chunks,
        embeddings=embeddings,
        metadata=[{"source": "demo", "chunk_index": i} for i in range(len(chunks))],
    )

    print(f"Vector store has {store.count} documents\n")

    # Search!
    query = "What is RAG?"
    print(f'Searching for: "{query}"\n')
    query_emb = generate_embedding(query, tracker=tracker)
    results = store.search(query_emb, k=3)

    for r in results:
        print(f"  Rank {r['rank']} (distance: {r['distance']:.4f}):")
        print(f"    {r['text'][:80]}...")
        print()
