import pytest
from pydantic import ValidationError

from app.core.models import AccessLevel, Document, SourceType


def test_document_valid_construction():
    """A Document with all required fields and valid enums should construct cleanly."""
    doc = Document(
        source_path="data/public_handbook.md",
        source_type=SourceType.MARKDOWN,
        access_level=AccessLevel.PUBLIC,
        title="Employee Handbook",
        content="Office hours are 9 to 6.",
    )

    assert doc.access_level == AccessLevel.PUBLIC
    assert doc.source_type == SourceType.MARKDOWN
    assert doc.doc_id  # auto-generated UUID string, should not be empty
    assert doc.ingested_at is not None


def test_document_rejects_invalid_access_level():
    """An invalid access_level string must fail validation, not silently pass through."""
    with pytest.raises(ValidationError):
        Document(
            source_path="data/some_file.md",
            source_type=SourceType.MARKDOWN,
            access_level="not_a_real_level",  # invalid on purpose
            title="Bad Doc",
            content="content",
        )


def test_document_rejects_invalid_source_type():
    """An invalid source_type must fail validation."""
    with pytest.raises(ValidationError):
        Document(
            source_path="data/some_file.xyz",
            source_type="carrier_pigeon",  # invalid on purpose
            access_level=AccessLevel.INTERNAL,
            title="Bad Doc",
            content="content",
        )


def test_each_document_gets_unique_id():
    """Two separately constructed Documents must not share a doc_id."""
    doc1 = Document(
        source_path="a.md",
        source_type=SourceType.MARKDOWN,
        access_level=AccessLevel.PUBLIC,
        title="A",
        content="a",
    )
    doc2 = Document(
        source_path="b.md",
        source_type=SourceType.MARKDOWN,
        access_level=AccessLevel.PUBLIC,
        title="B",
        content="b",
    )

    assert doc1.doc_id != doc2.doc_id