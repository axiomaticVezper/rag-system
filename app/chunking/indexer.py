"""
Qdrant indexing: takes Chunks, embeds them, and stores vectors + metadata
(payload) in a Qdrant collection. The payload is what enables RBAC-filtered
retrieval in Phase 3 -- Qdrant can filter on access_level before/during
vector search.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chunking.embedder import EMBEDDING_DIMENSION, embed_texts
from app.core.models import Chunk

COLLECTION_NAME = "rag_chunks"
QDRANT_URL = "http://localhost:6333"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection_exists() -> None:
    """Create the rag_chunks collection if it doesn't already exist. Safe to call repeatedly."""
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )


def index_chunks(chunks: list[Chunk]) -> int:
    """
    Embed and upload a list of Chunks into Qdrant.

    Returns the number of chunks indexed. Each chunk's full metadata
    (doc_id, access_level, doc_title, chunk_index, content) is stored
    as the point's payload, alongside its embedding vector.
    """
    if not chunks:
        return 0

    ensure_collection_exists()
    client = _get_client()

    texts = [chunk.content for chunk in chunks]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "access_level": chunk.access_level.value,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)