import uuid

import pytest

from app.chunking.indexer import (
    COLLECTION_NAME,
    _get_client,
    index_chunks,
    reset_collection,
)
from app.core.models import AccessLevel, Chunk


def _make_test_chunk(
    content: str, access_level: AccessLevel = AccessLevel.PUBLIC
) -> Chunk:
    return Chunk(
        doc_id=str(uuid.uuid4()),
        doc_title="Test Document",
        access_level=access_level,
        chunk_index=0,
        content=content,
        token_count=10,
    )


@pytest.fixture(autouse=True)
def clean_collection_after_test():
    """
    Reset the collection before and after every test in this file.
    This ensures test chunks never leak into the live collection and
    corrupt other tests or the demo script.
    After all tests in this file complete, repopulate with real sample
    data so retrieval tests still work.
    """
    reset_collection()
    yield
    reset_collection()
    # Repopulate with real data so the collection is never left empty.
    from app.core.models import AccessLevel
    from app.ingestion.pipeline import ingest_folder
    from app.chunking.chunker import chunk_document
    from app.chunking.indexer import index_chunks

    access_levels = {
        "public_handbook.md": AccessLevel.PUBLIC,
        "internal_engineering.html": AccessLevel.INTERNAL,
        "confidential_finance.pdf": AccessLevel.CONFIDENTIAL,
    }
    documents = ingest_folder("data", access_levels)
    for doc in documents:
        chunks = chunk_document(doc)
        index_chunks(chunks)


def test_index_chunks_stores_points_in_qdrant():
    """Indexed chunks should be retrievable from Qdrant with correct payload."""
    chunk = _make_test_chunk("Office hours are 9 to 6, Monday through Friday.")

    count = index_chunks([chunk])
    assert count == 1

    client = _get_client()
    retrieved = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[chunk.chunk_id],
        with_vectors=True,
    )

    assert len(retrieved) == 1
    point = retrieved[0]
    assert point.payload["access_level"] == "public"
    assert point.payload["content"] == chunk.content
    assert len(point.vector) == 384


def test_index_empty_list_returns_zero():
    """Indexing an empty list should be a safe no-op."""
    count = index_chunks([])
    assert count == 0