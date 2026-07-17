"""
Core data models for the RAG system.

These models define the internal, normalized representation that every
document flows into, regardless of its original source format (PDF, HTML,
Markdown, etc). Downstream stages (chunking, indexing, retrieval) all
depend on this shape rather than on any particular file format.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AccessLevel(str, Enum):
    """
    Access control tiers for RBAC-aware retrieval.

    Ordered from least to most restrictive. A user's clearance determines
    which documents can ever be returned to them during retrieval (Phase 3).
    Using an enum (instead of a raw string) means an invalid value like
    "Public" or "internel" fails immediately at ingestion time, rather than
    silently creating a security gap later.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class SourceType(str, Enum):
    """The original file format a document was ingested from."""

    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"


class Document(BaseModel):
    """
    Normalized representation of an ingested source document.

    Every loader (PDF, HTML, Markdown) must produce a Document with this
    exact shape. This is the single contract that chunking, embedding,
    and retrieval code depend on.
    """

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    source_type: SourceType
    access_level: AccessLevel
    title: str
    content: str
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {
        "use_enum_values": False,
    }
class Chunk(BaseModel):
    """
    A single retrievable unit produced by splitting a Document's content.

    Carries doc_id and access_level forward from its parent Document. This
    is how RBAC survives chunking: at retrieval time, we filter directly on
    each chunk's own access_level rather than re-looking-up its parent.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    doc_title: str
    access_level: AccessLevel
    chunk_index: int
    content: str
    token_count: int
class Answer(BaseModel):
    """
    Structured output from the generation step.

    citations refers to the index (0-based) of chunks in the list that was
    passed to the generator -- e.g. citations=[0, 2] means the answer draws
    on the 1st and 3rd retrieved chunks. This lets the caller map back to
    the actual source documents for display.
    """

    answer: str
    citations: list[int] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)