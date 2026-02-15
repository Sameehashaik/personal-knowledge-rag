# rag_chain.py
# Ties the whole pipeline together:
#   load file -> chunk -> embed -> store -> (user asks question) -> embed query
#   -> vector search -> build prompt with relevant chunks -> LLM answers -> done
# This is the only module the UI (app.py) needs to import.

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

# gpt-4o-mini for the chat completion (cheap and fast enough for RAG answers)
CHAT_MODEL = "gpt-4o-mini"


class RAGChain:
    """Full ingest-and-query pipeline. Feed it docs, ask it questions."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.vector_store = VectorStore(dimension=1536)
        self.tracker = CostTracker(log_file="project1_costs.json")
        self.documents: list[str] = []  # filenames we've ingested so far

    def add_document(self, file_path: str) -> dict:
        """
        Ingest one file: load -> chunk -> embed -> store.
        Returns a summary dict (filename, chunk count, char count, total in store).
        """
        text = load_document(file_path)
        filename = Path(file_path).name

        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        metadata_list = get_chunk_metadata(chunks)

        # tag every chunk with its source file so we can cite it later
        for meta in metadata_list:
            meta["source"] = filename

        # batch embed is one API call instead of N
        embeddings = generate_embeddings_batch(chunks, tracker=self.tracker)

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
        Assemble the system prompt + retrieved context + user question.
        The LLM is told to ONLY use context provided (no hallucinating).
        """
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
        End-to-end Q&A: embed the question, find nearest chunks,
        build a prompt, call the LLM, return answer + sources + cost.
        """
        if self.vector_store.count == 0:
            return {
                "answer": "No documents have been added yet. Please upload a document first.",
                "sources": [],
                "cost": 0.0,
                "chunks_used": 0,
            }

        query_embedding = generate_embedding(question, tracker=self.tracker)
        results = self.vector_store.search(query_embedding, k=k)

        # filter by distance — anything above 1.2 is probably noise;
        # but always keep at least the top-1 so the LLM can say "I don't know"
        RELEVANCE_THRESHOLD = 1.2
        relevant_results = [r for r in results if r["distance"] < RELEVANCE_THRESHOLD]
        results_for_prompt = relevant_results or results[:1]

        prompt = self.build_rag_prompt(question, results_for_prompt)

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # low temp = more factual, less creative
        )

        answer = response.choices[0].message.content

        usage = response.usage
        cost = self.tracker.track_call(
            model="gpt-4o-mini",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            description=f"RAG query: {question[:50]}",
        )

        # only surface chunks that actually passed the relevance filter
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


# --- end-to-end demo ---
if __name__ == "__main__":
    print("=== RAG Chain Demo ===\n")

    rag = RAGChain()

    sample_path = str(Path(__file__).parent.parent / "data" / "sample1.txt")
    print(f"Adding document: {sample_path}")
    summary = rag.add_document(sample_path)
    print(f"  -> {summary['chunks']} chunks from {summary['filename']}\n")

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
