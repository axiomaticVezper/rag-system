import uuid

import pytest

from app.chunking.indexer import COLLECTION_NAME, _get_client, index_chunks
from app.core.models import AccessLevel, Chunk


def _make_test_chunk(content: str, access_level: AccessLevel = AccessLevel.PUBLIC) -> Chunk:
    return Chunk(
        doc_id=str(uuid.uuid4()),
        doc_title="Test Document",
        access_level=access_level,
        chunk_index=0,
        content=content,
        token_count=10,
    )


def test_index_chunks_stores_points_in_qdrant():
    """Indexed chunks should be retrievable from Qdrant with correct payload."""
    chunk = _make_test_chunk("Office hours are 9 to 6, Monday through Friday.")

    count = index_chunks([chunk])
    assert count == 1

    client = _get_client()
    retrieved = client.retrieve(collection_name=COLLECTION_NAME, ids=[chunk.chunk_id], with_vectors=True)

    assert len(retrieved) == 1
    point = retrieved[0]
    assert point.payload["access_level"] == "public"
    assert point.payload["content"] == chunk.content
    assert len(point.vector) == 384


def test_index_empty_list_returns_zero():
    """Indexing an empty list should be a safe no-op."""
    count = index_chunks([])
    assert count == 0