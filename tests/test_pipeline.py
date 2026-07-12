import pytest

from app.core.models import AccessLevel, SourceType
from app.ingestion.pipeline import ingest_folder

ACCESS_LEVELS = {
    "public_handbook.md": AccessLevel.PUBLIC,
    "internal_engineering.html": AccessLevel.INTERNAL,
    "confidential_finance.pdf": AccessLevel.CONFIDENTIAL,
}


def test_ingest_folder_loads_all_three_documents():
    """All three sample documents should load with the correct access levels."""
    documents = ingest_folder("data", ACCESS_LEVELS)

    assert len(documents) == 3

    by_source_type = {doc.source_type: doc for doc in documents}
    assert SourceType.MARKDOWN in by_source_type
    assert SourceType.HTML in by_source_type
    assert SourceType.PDF in by_source_type

    assert by_source_type[SourceType.MARKDOWN].access_level == AccessLevel.PUBLIC
    assert by_source_type[SourceType.HTML].access_level == AccessLevel.INTERNAL
    assert by_source_type[SourceType.PDF].access_level == AccessLevel.CONFIDENTIAL


def test_ingest_folder_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        ingest_folder("data_does_not_exist", ACCESS_LEVELS)


def test_ingest_folder_missing_access_level_raises():
    """A file present in the folder but absent from access_levels must fail loudly."""
    incomplete_levels = {
        "public_handbook.md": AccessLevel.PUBLIC,
        "internal_engineering.html": AccessLevel.INTERNAL,
        # confidential_finance.pdf deliberately omitted
    }

    with pytest.raises(ValueError, match="No access_level specified"):
        ingest_folder("data", incomplete_levels)