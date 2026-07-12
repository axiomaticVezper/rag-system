"""
Loader for PDF files.

Uses pypdf to extract text page-by-page. PDFs generated from Word/plain
text layouts extract cleanly; complex multi-column layouts or scanned
(image-based) PDFs may need OCR, which is out of scope here.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.core.models import AccessLevel, Document, SourceType


def load_pdf_document(file_path: str, access_level: AccessLevel) -> Document:
    """
    Load a .pdf file from disk into a normalized Document.

    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the PDF has no extractable text (e.g. scanned images).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(str(path))

    page_texts = [page.extract_text() or "" for page in reader.pages]
    content = "\n".join(page_texts).strip()

    if not content:
        raise ValueError(
            f"No extractable text found in PDF: {file_path}. "
            "It may be a scanned/image-based PDF requiring OCR."
        )

    return Document(
        source_path=str(path),
        source_type=SourceType.PDF,
        access_level=access_level,
        title=path.stem.replace("_", " ").title(),
        content=content,
    )