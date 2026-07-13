from app.core.models import AccessLevel, Chunk


def test_chunk_valid_construction():
    """A Chunk should construct cleanly and retain its parent document's identity."""
    chunk = Chunk(
        doc_id="some-doc-uuid",
        doc_title="Employee Handbook",
        access_level=AccessLevel.PUBLIC,
        chunk_index=0,
        content="Office hours are 9 to 6.",
        token_count=7,
    )

    assert chunk.doc_id == "some-doc-uuid"
    assert chunk.access_level == AccessLevel.PUBLIC
    assert chunk.chunk_index == 0
    assert chunk.chunk_id  # auto-generated, should not be empty


def test_each_chunk_gets_unique_id():
    """Two chunks from the same document must still have distinct chunk_ids."""
    chunk1 = Chunk(
        doc_id="doc-1", doc_title="Doc", access_level=AccessLevel.INTERNAL,
        chunk_index=0, content="first part", token_count=2,
    )
    chunk2 = Chunk(
        doc_id="doc-1", doc_title="Doc", access_level=AccessLevel.INTERNAL,
        chunk_index=1, content="second part", token_count=2,
    )

    assert chunk1.chunk_id != chunk2.chunk_id
    assert chunk1.doc_id == chunk2.doc_id  # same parent document