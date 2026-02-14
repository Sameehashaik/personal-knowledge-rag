"""
RAG Chain - The complete pipeline that ties everything together.

This is the brain of the system. When a user asks a question:
  1. Load & chunk the document        (document_loader + chunker)
  2. Embed the chunks & store them    (embeddings + vector_store)
  3. Embed the question               (embeddings)
  4. Find relevant chunks             (vector_store.search)
  5. Build a prompt with the chunks   (prompt engineering)
  6. Ask the LLM to answer            (OpenAI chat API)
  7. Return the answer with sources   (citation)
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from src.document_loader import load_document
from src.chunker import chunk_text, get_chunk_metadata
from src.embeddings import generate_embedding, generate_embeddings_batch
from src.vector_store import VectorStore

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker

load_dotenv()

# The LLM we'll use for answering questions
CHAT_MODEL = "gpt-4o-mini"


class RAGChain:
    """
    Complete RAG pipeline: ingest documents, answer questions with sources.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the RAG chain.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.vector_store = VectorStore(dimension=1536)
        self.tracker = CostTracker(log_file="project1_costs.json")
        self.documents: list[str] = []  # Track which files have been added

    def add_document(self, file_path: str) -> dict:
        """
        Ingest a document into the RAG system.

        Steps: load -> chunk -> embed -> store

        Args:
            file_path: Path to a .txt or .pdf file.

        Returns:
            Summary dict with chunk count and file info.
        """
        # Step 1: Load the document
        text = load_document(file_path)
        filename = Path(file_path).name

        # Step 2: Chunk the text
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        metadata_list = get_chunk_metadata(chunks)

        # Add source filename to each chunk's metadata
        for meta in metadata_list:
            meta["source"] = filename

        # Step 3: Generate embeddings for all chunks (batch = efficient)
        embeddings = generate_embeddings_batch(chunks, tracker=self.tracker)

        # Step 4: Store in vector database
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata=metadata_list,
        )

        self.documents.append(filename)

        return {
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
            "total_chunks_in_store": self.vector_store.count,
        }

    def build_rag_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        Build the prompt that we send to the LLM.

        This is where prompt engineering happens. We give the LLM:
        - Clear instructions on how to behave
        - The retrieved context chunks
        - The user's question

        Args:
            question: The user's question.
            chunks: Retrieved chunks from the vector store.

        Returns:
            The formatted prompt string.
        """
        # Format the context chunks with source labels
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
- Only use information from the context below to answer.
- If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Cite which source(s) you used by referencing [Source N].
- Be concise and direct.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        return prompt

    def query(self, question: str, k: int = 1) -> dict:
        """
        Ask a question and get an answer based on stored documents.

        Args:
            question: The user's question.
            k: Number of relevant chunks to retrieve.

        Returns:
            Dict with: answer, sources, cost, chunks_used.
        """
        if self.vector_store.count == 0:
            return {
                "answer": "No documents have been added yet. Please upload a document first.",
                "sources": [],
                "cost": 0.0,
                "chunks_used": 0,
            }

        # Step 1: Embed the question
        query_embedding = generate_embedding(question, tracker=self.tracker)

        # Step 2: Find relevant chunks
        results = self.vector_store.search(query_embedding, k=k)

        # Filter out irrelevant results (high distance = poor match).
        # If nothing passes the threshold, still send the best result
        # so the LLM can decide to say "I don't know."
        RELEVANCE_THRESHOLD = 1.2
        relevant_results = [r for r in results if r["distance"] < RELEVANCE_THRESHOLD]
        results_for_prompt = relevant_results or results[:1]

        # Step 3: Build the prompt
        prompt = self.build_rag_prompt(question, results_for_prompt)

        # Step 4: Ask the LLM
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Low temperature = more focused/factual answers
        )

        answer = response.choices[0].message.content

        # Track LLM cost
        usage = response.usage
        cost = self.tracker.track_call(
            model="gpt-4o-mini",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            description=f"RAG query: {question[:50]}",
        )

        # Only show relevant sources to the user (don't display irrelevant chunks)
        sources = []
        for r in relevant_results:
            sources.append({
                "source": r["metadata"].get("source", "unknown"),
                "chunk_index": r["metadata"].get("chunk_index", -1),
                "distance": r["distance"],
                "preview": r["text"],
            })

        return {
            "answer": answer,
            "sources": sources,
            "cost": cost,
            "chunks_used": len(relevant_results),
        }


# End-to-end demo
if __name__ == "__main__":
    print("=== RAG Chain Demo ===\n")

    rag = RAGChain()

    # Add our sample document
    sample_path = str(Path(__file__).parent.parent / "data" / "sample1.txt")
    print(f"Adding document: {sample_path}")
    summary = rag.add_document(sample_path)
    print(f"  -> {summary['chunks']} chunks from {summary['filename']}\n")

    # Ask questions
    questions = [
        "What is RAG?",
        "How does cosine similarity work?",
        "What is the best pizza topping?",
    ]

    for q in questions:
        print(f"Q: {q}")
        result = rag.query(q)
        print(f"A: {result['answer']}\n")
        print(f"   Sources: {[s['source'] for s in result['sources']]}")
        print(f"   Cost: ${result['cost']:.6f}\n")
        print("-" * 60 + "\n")
