import pytest

from app.core.models import AccessLevel, SourceType
from app.ingestion.text_loader import load_text_document


def test_load_public_handbook():
    """Loading the sample public handbook should extract the title from its '#' heading."""
    doc = load_text_document(
        file_path="data/public_handbook.md",
        access_level=AccessLevel.PUBLIC,
    )

    assert doc.source_type == SourceType.MARKDOWN
    assert doc.access_level == AccessLevel.PUBLIC
    assert doc.title == "Employee Handbook (Public Excerpt)"
    assert "Office Hours" in doc.content
    assert doc.source_path.endswith("public_handbook.md")


def test_load_missing_file_raises():
    """A nonexistent path must raise FileNotFoundError, not fail silently."""
    with pytest.raises(FileNotFoundError):
        load_text_document(
            file_path="data/does_not_exist.md",
            access_level=AccessLevel.PUBLIC,
        )   