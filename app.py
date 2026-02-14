"""
Streamlit UI - Web interface for the Personal Knowledge RAG system.

Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from src.rag_chain import RAGChain

# --- Page Config ---
st.set_page_config(
    page_title="Personal Knowledge RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Personal Knowledge RAG")
st.caption("Upload your documents, then ask questions about them.")


# --- Session State ---
# Streamlit reruns the entire script on every interaction.
# session_state persists data across those reruns.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain()
    st.session_state.uploaded_files = []
    st.session_state.total_cost = 0.0
    st.session_state.query_count = 0


# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"],
        help="Supported formats: TXT, PDF",
    )

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save uploaded file to a temp location so our loader can read it.
            # Use the original filename so metadata tracks the real name.
            original_name = uploaded_file.name
            suffix = Path(original_name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix=Path(original_name).stem + "_"
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                summary = st.session_state.rag_chain.add_document(tmp_path)

                # Override the temp filename with the original upload name
                # in both our tracking list and the vector store metadata
                st.session_state.uploaded_files.append(original_name)
                st.session_state.rag_chain.documents[-1] = original_name
                for meta in st.session_state.rag_chain.vector_store.metadata:
                    if meta.get("source") == summary["filename"]:
                        meta["source"] = original_name

                st.success(
                    f"Added **{original_name}** — "
                    f"{summary['chunks']} chunks, "
                    f"{summary['characters']:,} characters"
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                os.unlink(tmp_path)  # Clean up temp file

    # Show uploaded documents
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.write(f"✅ {fname}")
        st.caption(
            f"Total chunks in store: "
            f"{st.session_state.rag_chain.vector_store.count}"
        )

    # Cost tracker — use st.sidebar placeholders so they update after queries
    st.divider()
    st.subheader("💰 Cost Tracker")
    cost_placeholder = st.empty()
    queries_placeholder = st.empty()


# --- Main Area: Q&A ---
if not st.session_state.uploaded_files:
    st.info("👈 Upload a document in the sidebar to get started!")
else:
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is RAG?",
    )

    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain.query(question)

        # Update cost tracking
        if result["cost"]:
            st.session_state.total_cost += result["cost"]
        st.session_state.query_count += 1

        # Display the answer
        st.markdown("### Answer")
        st.write(result["answer"])

        # Display sources in an expander
        with st.expander(f"📋 Sources ({result['chunks_used']} chunks used)"):
            for source in result["sources"]:
                st.markdown(
                    f"**{source['source']}** (chunk {source['chunk_index']}, "
                    f"distance: {source['distance']:.4f})"
                )
                st.text(source["preview"])
                st.divider()

        # Show cost for this query
        if result["cost"]:
            st.caption(f"Query cost: ${result['cost']:.6f}")

# Update the sidebar cost metrics AFTER any query has run
# This ensures they reflect the latest values
cost_placeholder.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
queries_placeholder.metric("Queries Made", st.session_state.query_count)
