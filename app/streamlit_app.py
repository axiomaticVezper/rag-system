"""
Streamlit frontend for the RAG system.

Calls the FastAPI backend over HTTP -- not importing app code directly --
so this works correctly whether running locally or in Docker Compose where
frontend and backend are separate containers.
"""

from __future__ import annotations

import httpx
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 RAG System")
st.caption("Retrieval-Augmented Generation with RBAC access control")

with st.sidebar:
    st.header("Settings")
    clearance = st.selectbox(
        "Your clearance level",
        options=["public", "internal", "confidential"],
        index=0,
        help="Controls which documents you can retrieve answers from",
    )
    st.divider()
    st.markdown("**Clearance levels:**")
    st.markdown("- `public` — general company info")
    st.markdown("- `internal` — engineering wiki, internal docs")
    st.markdown("- `confidential` — financial reports, sensitive data")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the office hours?",
)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = httpx.post(
                f"{API_URL}/ask",
                json={"question": question, "clearance": clearance},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            st.success("Answer")
            st.write(data["answer"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{data['confidence_score']:.0%}")
            with col2:
                st.metric("Citations", str(data["citations"]))

            st.divider()
            st.caption(f"Clearance used: `{data['clearance_used']}`")

        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
        except httpx.ConnectError:
            st.error(
                "Cannot connect to the API backend. "
                "Make sure the FastAPI server is running: "
                "`uvicorn app.main:app --host 0.0.0.0 --port 8000`"
            )

with st.expander("Re-index documents"):
    st.caption("Run this if you've added new documents to the data/ folder.")
    if st.button("Run ingestion pipeline"):
        with st.spinner("Ingesting and indexing documents..."):
            try:
                response = httpx.post(f"{API_URL}/ingest", timeout=300.0)
                response.raise_for_status()
                data = response.json()
                st.success(
                    f"Indexed {data['chunks_indexed']} chunks "
                    f"from {data['documents_loaded']} documents."
                )
            except httpx.ConnectError:
                st.error("Cannot connect to the API backend.")