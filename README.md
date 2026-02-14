# Personal Knowledge RAG

A Retrieval-Augmented Generation (RAG) system built from scratch in Python. Upload your documents (TXT, PDF), and ask natural language questions — the system finds relevant passages and generates accurate, cited answers using OpenAI's API.

**Built as a learning project to deeply understand every component of the RAG pipeline.**

## Demo

![Demo](assets/Personal_knowledge_RAG_DEMO.gif)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                 │
│              Upload documents  ·  Ask questions              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  RAG Chain (rag_chain.py)                     │
│         Orchestrates the full pipeline end-to-end            │
└──┬──────────┬──────────┬──────────┬─────────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
│Document│ │ Text   │ │Embeddings│ │ Vector Store │
│ Loader │ │Chunker │ │(OpenAI)  │ │  (FAISS)     │
└────────┘ └────────┘ └──────────┘ └──────────────┘
```

### How a query flows through the system

```
User asks: "What is RAG?"
   │
   ├─ 1. Question → OpenAI Embeddings API → 1536-dim vector
   ├─ 2. Vector  → FAISS similarity search → Top-k relevant chunks
   ├─ 3. Chunks  → Prompt template with context + rules → Formatted prompt
   ├─ 4. Prompt  → GPT-4o-mini → Answer with [Source N] citations
   └─ 5. Answer + Sources + Cost → Returned to UI
```

## Key Concepts (Interview Reference)

### What is RAG?
RAG (Retrieval-Augmented Generation) gives an LLM access to external documents at query time. Instead of relying only on training data, the model receives relevant context and generates grounded answers. This reduces hallucination, keeps data private, and avoids expensive fine-tuning.

### Embeddings
Text is converted into 1536-dimensional float vectors using OpenAI's `text-embedding-3-small` model. Semantically similar text produces similar vectors — "artificial intelligence" and "machine learning" end up close together in vector space, while "pizza recipe" is far away. This is how the system understands meaning, not just keywords.

### Cosine Similarity vs L2 Distance
- **Cosine similarity**: Measures the angle between vectors. 1.0 = identical direction, 0.0 = perpendicular. Good for comparing meaning regardless of text length.
- **L2 (Euclidean) distance**: Measures straight-line distance. 0.0 = identical, higher = less similar. This is what FAISS `IndexFlatL2` uses.

### Chunking Strategy
Documents are split into ~1000-character chunks with 200-character overlap. Splitting happens on sentence boundaries (not mid-word). Overlap ensures information at chunk boundaries isn't lost. Trade-off: smaller chunks = more precise retrieval but less context per chunk.

### Vector Database (FAISS)
FAISS (Facebook AI Similarity Search) stores embedding vectors and finds the k-nearest neighbors to a query vector. We use `IndexFlatL2` (brute-force exact search) — fast enough for thousands of chunks. For millions, you'd switch to `IndexIVFFlat` (approximate search with clustering).

### Prompt Engineering
The RAG prompt includes strict rules:
- Only answer from provided context
- Cite sources using `[Source N]`
- Admit when information isn't available

Temperature is set to 0.3 (low = focused/factual, high = creative).

### Relevance Threshold
Retrieved chunks with L2 distance > 1.2 are filtered out. This prevents showing irrelevant sources when the user's question doesn't match any document content (e.g., typing "hi" won't display unrelated chunks).

## Project Structure

```
personal-knowledge-rag/
├── app.py                  # Streamlit web UI
├── cost_tracker.py         # API cost tracking utility
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── src/
│   ├── document_loader.py  # Load text from TXT and PDF files
│   ├── chunker.py          # Split text into overlapping chunks
│   ├── embeddings.py       # Generate OpenAI embeddings + cosine similarity
│   ├── vector_store.py     # FAISS-based vector database
│   └── rag_chain.py        # Complete RAG pipeline orchestrator
├── data/
│   ├── sample1.txt         # Sample doc: RAG concepts
│   └── sample2.txt         # Sample doc: Cloud computing
└── tests/
    ├── test_loader.py      # 10 tests
    ├── test_chunker.py     # 11 tests
    ├── test_embeddings.py  # 11 tests
    ├── test_vector_store.py# 11 tests
    └── test_rag_chain.py   # 10 tests
```

## Setup

### Prerequisites
- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repo
git clone https://github.com/Sameehashaik/personal-knowledge-rag.git
cd personal-knowledge-rag

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
# Edit .env and replace the placeholder with your key
```

### Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload a document, and start asking questions.

### Run Tests

```bash
# All tests (some hit OpenAI API — costs ~$0.01)
python -m pytest tests/ -v

# Only offline tests (free, no API calls)
python -m pytest tests/test_loader.py tests/test_chunker.py tests/test_vector_store.py -v
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | OpenAI `text-embedding-3-small` | Cheap ($0.02/1M tokens), 1536 dims, good quality |
| LLM | OpenAI `gpt-4o-mini` | Fast, cheap ($0.15/$0.60 per 1M tokens), sufficient for Q&A |
| Vector DB | FAISS (`IndexFlatL2`) | Fast exact search, no server needed, runs locally |
| PDF parsing | PyPDF2 | Simple, well-documented Python library |
| Web UI | Streamlit | Python-native, no frontend code needed |
| Cost tracking | Custom `CostTracker` | Logs every API call to JSON, tracks spend per session |

## Cost Breakdown

| Operation | Approximate Cost |
|-----------|-----------------|
| Embed a document (~6000 chars) | ~$0.001 |
| Single query (embed + LLM) | ~$0.0002 |
| Full test suite | ~$0.01 |
| **Typical session (5 docs, 20 queries)** | **~$0.01-0.02** |

All costs are tracked in `project1_costs.json` and displayed in the Streamlit sidebar.

## What I Learned Building This

1. **File I/O & encoding** — Why UTF-8 matters and how `with` context managers prevent resource leaks
2. **Text chunking** — Sentence-boundary splitting, overlap for context preservation, chunk size trade-offs
3. **Embeddings** — How text becomes vectors, why 1536 dimensions, cosine similarity vs L2 distance
4. **Vector databases** — K-nearest neighbors, FAISS index types, brute-force vs approximate search
5. **Prompt engineering** — Grounding LLM answers in context, citation rules, temperature tuning
6. **RAG pipeline** — End-to-end flow from document upload to cited answer
7. **Streamlit** — Session state, rerun model, file uploads, reactive UI from Python
8. **Cost awareness** — Tracking every API call, understanding token pricing, optimizing batch calls

## Limitations & Future Improvements

- **No persistence** — Vector store resets on app restart (could add save/load to disk)
- **No PDF table/image support** — PyPDF2 only extracts text, not structured content
- **Single-user** — Streamlit session state is per-tab, no multi-user support
- **No conversation memory** — Each question is independent (could add chat history)
- **Fixed chunk size** — Could experiment with semantic chunking (split by topic, not character count)
