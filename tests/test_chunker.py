from app.chunking.chunker import chunk_document, _token_count
from app.core.models import AccessLevel, Document, SourceType


def _make_document(content: str) -> Document:
    return Document(
        source_path="data/test.md",
        source_type=SourceType.MARKDOWN,
        access_level=AccessLevel.INTERNAL,
        title="Test Doc",
        content=content,
    )


def test_short_document_produces_single_chunk():
    """A document well under the token limit should produce exactly one chunk."""
    doc = _make_document("This is a short paragraph.\n\nAnd a second one.")
    chunks = chunk_document(doc, token_limit=512, overlap_tokens=50)

    assert len(chunks) == 1
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].access_level == AccessLevel.INTERNAL
    assert chunks[0].chunk_index == 0


def test_long_document_produces_multiple_chunks():
    """A document exceeding the token limit should split into more than one chunk."""
    # Build a document well over 512 tokens using repeated distinct paragraphs.
    paragraphs = [f"This is paragraph number {i} with some extra padding words." for i in range(80)]
    doc = _make_document("\n\n".join(paragraphs))

    chunks = chunk_document(doc, token_limit=100, overlap_tokens=20)

    assert len(chunks) > 1
    # Every chunk must respect the token limit (allowing small overlap slack).
    for chunk in chunks:
        assert chunk.token_count <= 150  # limit + reasonable overlap slack
    # chunk_index must be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_all_chunks_preserve_parent_metadata():
    """Every chunk from one document must carry the same doc_id and access_level."""
    paragraphs = [f"Paragraph {i} content here." for i in range(50)]
    doc = _make_document("\n\n".join(paragraphs))

    chunks = chunk_document(doc, token_limit=80, overlap_tokens=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == doc.doc_id
        assert chunk.access_level == doc.access_level
        assert chunk.doc_title == doc.title


def test_overlap_carries_content_between_chunks():
    """The tail of one chunk should reappear at the start of the next (the overlap)."""
    paragraphs = [f"Unique sentence marker number {i} here." for i in range(30)]
    doc = _make_document("\n\n".join(paragraphs))

    chunks = chunk_document(doc, token_limit=60, overlap_tokens=15)

    assert len(chunks) > 1
    # The end of chunk 0 and the start of chunk 1 should share some text
    # because of the overlap mechanism.
    end_of_first = chunks[0].content[-40:]
    start_of_second = chunks[1].content[:80]
    # At least one word from the tail of chunk 0 should appear in chunk 1's start.
    tail_words = set(end_of_first.split())
    start_words = set(start_of_second.split())
    assert tail_words & start_words  # non-empty intersection