import pytest

from app.core.models import AccessLevel, SourceType
from app.ingestion.html_loader import load_html_document


def test_load_internal_engineering_doc():
    """Loading the sample HTML file should extract the <title> and clean text content."""
    doc = load_html_document(
        file_path="data/internal_engineering.html",
        access_level=AccessLevel.INTERNAL,
    )

    assert doc.source_type == SourceType.HTML
    assert doc.access_level == AccessLevel.INTERNAL
    assert doc.title == "Engineering Wiki - Deployment Guide"
    assert "Deployment Process" in doc.content
    assert "<h2>" not in doc.content  # tags must be stripped, not just visible as text
    assert "<script>" not in doc.content


def test_load_missing_html_raises():
    with pytest.raises(FileNotFoundError):
        load_html_document(
            file_path="data/does_not_exist.html",
            access_level=AccessLevel.INTERNAL,
        )
        