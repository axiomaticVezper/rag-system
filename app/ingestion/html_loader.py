"""
Loader for HTML files.

Uses BeautifulSoup to strip markup, scripts, and styles, leaving clean
text suitable for chunking and embedding. Title is taken from <title>,
falling back to the first <h1>, falling back to the filename.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.core.models import AccessLevel, Document, SourceType


def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    return fallback


def load_html_document(file_path: str, access_level: AccessLevel) -> Document:
    """
    Load an .html file from disk into a normalized Document.

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")

    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove elements that would pollute extracted text
    for tag in soup(["script", "style"]):
        tag.decompose()

    title = _extract_title(soup, fallback=path.stem)

    # get_text with a separator keeps paragraphs from running into each other
    content = soup.get_text(separator="\n", strip=True)

    return Document(
        source_path=str(path),
        source_type=SourceType.HTML,
        access_level=access_level,
        title=title,
        content=content,
    )