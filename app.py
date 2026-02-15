# app.py
# Streamlit front-end for the RAG system.
# Run with: streamlit run app.py

import streamlit as st
import tempfile
import os
from pathlib import Path
from src.rag_chain import RAGChain

st.set_page_config(
    page_title="Personal Knowledge RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Personal Knowledge RAG")
st.caption("Upload your documents, then ask questions about them.")

# streamlit reruns the whole script on every click/keystroke,
# so anything that should survive reruns goes in session_state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain()
    st.session_state.uploaded_files = []
    st.session_state.total_cost = 0.0
    st.session_state.query_count = 0


# ---- sidebar: file upload + doc list + cost display ----
with st.sidebar:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"],
        help="Supported formats: TXT, PDF",
    )

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # write to temp file because document_loader reads from disk
            original_name = uploaded_file.name
            suffix = Path(original_name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix=Path(original_name).stem + "_"
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                summary = st.session_state.rag_chain.add_document(tmp_path)

                # replace the temp filename with the real upload name everywhere
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
                os.unlink(tmp_path)

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.write(f"✅ {fname}")
        st.caption(
            f"Total chunks in store: "
            f"{st.session_state.rag_chain.vector_store.count}"
        )

    # placeholders so cost numbers update after a query runs below
    st.divider()
    st.subheader("💰 Cost Tracker")
    cost_placeholder = st.empty()
    queries_placeholder = st.empty()


# ---- main area: question input + answer display ----
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

        if result["cost"]:
            st.session_state.total_cost += result["cost"]
        st.session_state.query_count += 1

        st.markdown("### Answer")
        st.write(result["answer"])

        with st.expander(f"📋 Sources ({result['chunks_used']} chunks used)"):
            for source in result["sources"]:
                st.markdown(
                    f"**{source['source']}** (chunk {source['chunk_index']}, "
                    f"distance: {source['distance']:.4f})"
                )
                st.text(source["preview"])
                st.divider()

        if result["cost"]:
            st.caption(f"Query cost: ${result['cost']:.6f}")

# refresh sidebar cost metrics now that the query (if any) has finished
cost_placeholder.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
queries_placeholder.metric("Queries Made", st.session_state.query_count)
