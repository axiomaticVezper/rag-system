"""
Semantic (paragraph-aware) chunking of Document content into Chunks.

Strategy:
1. Split content into paragraphs.
2. Greedily group consecutive paragraphs into a chunk until adding the
   next paragraph would exceed the token budget (default 512).
3. Carry forward ~50 tokens of overlap from the end of one chunk into the
   start of the next, so facts near a boundary aren't lost from context.
4. If a single paragraph alone exceeds the token budget, split it by
   sentence as a fallback (rare with well-formed source docs).
"""

from __future__ import annotations

import re

import tiktoken

from app.core.models import Chunk, Document

DEFAULT_CHUNK_TOKEN_LIMIT = 512
DEFAULT_OVERLAP_TOKENS = 50

# cl100k_base is the tokenizer used by GPT-3.5/4-family models; it's a
# reasonable, widely-used approximation for "how big is this text" even
# though we aren't calling OpenAI for embeddings.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_into_paragraphs(content: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", content)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_paragraph_by_sentence(paragraph: str, token_limit: int) -> list[str]:
    """Fallback for a single paragraph that alone exceeds the token limit."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if _token_count(candidate) > token_limit and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def _take_overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` worth of `text`, decoded back to a string."""
    token_ids = _ENCODING.encode(text)
    tail_ids = token_ids[-overlap_tokens:]
    return _ENCODING.decode(tail_ids)


def chunk_document(
    document: Document,
    token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Split a Document's content into a list of Chunks, preserving doc_id
    and access_level on every chunk.
    """
    paragraphs = _split_into_paragraphs(document.content)

    # Expand any paragraph that alone exceeds the token limit.
    units: list[str] = []
    for paragraph in paragraphs:
        if _token_count(paragraph) > token_limit:
            units.extend(_split_paragraph_by_sentence(paragraph, token_limit))
        else:
            units.append(paragraph)

    chunks: list[Chunk] = []
    current_text = ""
    chunk_index = 0

    for unit in units:
        candidate = f"{current_text}\n\n{unit}".strip() if current_text else unit

        if _token_count(candidate) > token_limit and current_text:
            # Close out the current chunk.
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    access_level=document.access_level,
                    chunk_index=chunk_index,
                    content=current_text,
                    token_count=_token_count(current_text),
                )
            )
            chunk_index += 1

            # Start the next chunk with overlap from the end of the last one.
            overlap_text = _take_overlap_tail(current_text, overlap_tokens)
            current_text = f"{overlap_text}\n\n{unit}".strip()
        else:
            current_text = candidate

    if current_text:
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                doc_title=document.title,
                access_level=document.access_level,
                chunk_index=chunk_index,
                content=current_text,
                token_count=_token_count(current_text),
            )
        )

    return chunks