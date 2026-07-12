import pytest

from app.core.models import AccessLevel, SourceType
from app.ingestion.pdf_loader import load_pdf_document


def test_load_confidential_finance_pdf():
    """Loading the sample PDF should extract text content and set the correct access level."""
    doc = load_pdf_document(
        file_path="data/confidential_finance.pdf",
        access_level=AccessLevel.CONFIDENTIAL,
    )

    assert doc.source_type == SourceType.PDF
    assert doc.access_level == AccessLevel.CONFIDENTIAL
    assert "Revenue" in doc.content
    assert len(doc.content) > 100  # sanity check that real text was extracted


def test_load_missing_pdf_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf_document(
            file_path="data/does_not_exist.pdf",
            access_level=AccessLevel.CONFIDENTIAL,
        )