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