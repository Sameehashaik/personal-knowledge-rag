# vector_store.py
# FAISS-backed vector DB — stores embeddings + original text + metadata side by side.
# Searches by meaning (L2 distance in embedding space) instead of keywords.

import json
import numpy as np
import faiss
from pathlib import Path


class VectorStore:
    """
    Wraps a FAISS flat-L2 index with parallel lists for the chunk text and metadata.
    save()/load() persist everything to a .faiss + .json file pair.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        # brute-force L2 — exact results, fine up to ~10k chunks
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict] = None,
    ) -> None:
        """Insert chunks + their vectors into the store. Metadata is optional."""
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(embeddings)} embeddings"
            )

        if metadata and len(metadata) != len(texts):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(metadata)} metadata entries"
            )

        # FAISS needs float32 numpy arrays
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)

        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

    def search(self, query_embedding: list[float], k: int = 3) -> list[dict]:
        """
        Return the k nearest chunks to query_embedding.
        Each result dict has: text, metadata, distance (L2), rank (1-indexed).
        Lower distance = more relevant.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)  # can't ask for more than we have

        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS pads with -1 when index is sparse
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
        """How many vectors are in the index right now."""
        return self.index.ntotal

    def save(self, filepath: str) -> None:
        """
        Write to disk as two files:
          <filepath>.faiss  — the FAISS index binary
          <filepath>.json   — texts + metadata + dimension
        """
        path = Path(filepath)

        faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        store_data = {
            "dimension": self.dimension,
            "texts": self.texts,
            "metadata": self.metadata,
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(store_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Reconstruct a VectorStore from a .faiss + .json file pair."""
        path = Path(filepath)

        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            store_data = json.load(f)

        store = cls(dimension=store_data["dimension"])
        store.texts = store_data["texts"]
        store.metadata = store_data["metadata"]
        store.index = faiss.read_index(str(path.with_suffix(".faiss")))

        return store


# --- quick demo: embed 5 chunks, store them, search ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.embeddings import generate_embeddings_batch, generate_embedding, cosine_similarity
    from cost_tracker import CostTracker

    tracker = CostTracker(log_file="project1_costs.json")

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

    query = "What is RAG?"
    print(f'Searching for: "{query}"\n')
    query_emb = generate_embedding(query, tracker=tracker)
    results = store.search(query_emb, k=3)

    for r in results:
        print(f"  Rank {r['rank']} (distance: {r['distance']:.4f}):")
        print(f"    {r['text'][:80]}...")
        print()
