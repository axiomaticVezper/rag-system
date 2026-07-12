"""
Ingestion pipeline orchestrator.

This is the single entry point for turning a folder of mixed-format files
into a list of normalized Document objects. It hides which loader handles
which file extension behind one function call.
"""

from __future__ import annotations

from pathlib import Path

from app.core.models import AccessLevel, Document
from app.ingestion.html_loader import load_html_document
from app.ingestion.pdf_loader import load_pdf_document
from app.ingestion.text_loader import load_text_document

# Maps file extensions to the loader function that handles them.
_LOADER_BY_EXTENSION = {
    ".md": load_text_document,
    ".txt": load_text_document,
    ".html": load_html_document,
    ".htm": load_html_document,
    ".pdf": load_pdf_document,
}


def ingest_folder(
    folder_path: str,
    access_levels: dict[str, AccessLevel],
) -> list[Document]:
    """
    Load every file in `folder_path` into a Document, using the correct
    loader based on file extension.

    Args:
        folder_path: directory containing the source files.
        access_levels: maps filename (e.g. "public_handbook.md") to the
            AccessLevel it should be tagged with. Every file in the folder
            must have an entry here, or ingestion fails loudly.

    Raises:
        FileNotFoundError: if folder_path does not exist.
        ValueError: if a file's extension is unsupported, or a file is
            missing from access_levels.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Ingestion folder not found: {folder_path}")

    documents: list[Document] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue  # skip subdirectories, if any

        filename = file_path.name
        extension = file_path.suffix.lower()

        loader = _LOADER_BY_EXTENSION.get(extension)
        if loader is None:
            raise ValueError(
                f"Unsupported file extension '{extension}' for file '{filename}'. "
                f"Supported extensions: {list(_LOADER_BY_EXTENSION.keys())}"
            )

        if filename not in access_levels:
            raise ValueError(
                f"No access_level specified for file '{filename}'. "
                "Every file must have an explicit access level assigned."
            )

        access_level = access_levels[filename]
        document = loader(str(file_path), access_level)
        documents.append(document)

    return documents