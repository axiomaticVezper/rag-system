"""
Loader for plain text and Markdown files.

Markdown/text files need no special parsing library since we treat the
raw content as-is. This is the simplest loader and establishes the pattern
every other loader (HTML, PDF) follows: given a file path and an access
level, return a normalized Document.
"""

from __future__ import annotations

from pathlib import Path

from app.core.models import AccessLevel, Document, SourceType


def _extract_title(content: str, fallback: str) -> str:
    """Use the first Markdown '# Heading' as the title, or fall back to the filename."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_text_document(file_path: str, access_level: AccessLevel) -> Document:
    """
    Load a .md or .txt file from disk into a normalized Document.

    Raises FileNotFoundError if the path does not exist, so ingestion
    fails loudly rather than silently skipping a missing file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text/Markdown file not found: {file_path}")

    content = path.read_text(encoding="utf-8")
    title = _extract_title(content, fallback=path.stem)

    source_type = SourceType.MARKDOWN if path.suffix.lower() == ".md" else SourceType.TEXT

    return Document(
        source_path=str(path),
        source_type=source_type,
        access_level=access_level,
        title=title,
        content=content,
    )