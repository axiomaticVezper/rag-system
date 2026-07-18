"""
FastAPI application entry point.

Exposes the RAG system as an HTTP API with three endpoints:
- GET  /health  — liveness check
- POST /ask     — question answering with RBAC-filtered hybrid retrieval
- POST /ingest  — trigger document ingestion and indexing
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chunking.chunker import chunk_document
from app.chunking.indexer import index_chunks, reset_collection
from app.core.models import AccessLevel, Answer
from app.generation.rag_pipeline import answer_question
from app.ingestion.pipeline import ingest_folder

app = FastAPI(
    title="RAG System API",
    description="Production-grade Retrieval-Augmented Generation with RBAC",
    version="1.0.0",
)

_ACCESS_LEVELS = {
    "public_handbook.md": AccessLevel.PUBLIC,
    "internal_engineering.html": AccessLevel.INTERNAL,
    "confidential_finance.pdf": AccessLevel.CONFIDENTIAL,
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    clearance: str = Field(
        default="public",
        description="User clearance level: public, internal, or confidential",
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    confidence_score: float
    clearance_used: str


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-system"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        clearance = AccessLevel(request.clearance.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clearance level '{request.clearance}'. "
                   f"Must be one of: public, internal, confidential",
        )

    result: Answer = answer_question(request.question, clearance)

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        confidence_score=result.confidence_score,
        clearance_used=clearance.value,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    reset_collection()
    documents = ingest_folder("data", _ACCESS_LEVELS)

    total_chunks = 0
    for doc in documents:
        chunks = chunk_document(doc)
        total_chunks += index_chunks(chunks)

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_indexed=total_chunks,
    )