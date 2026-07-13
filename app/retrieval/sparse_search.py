"""
Sparse (BM25 keyword) search over indexed chunks.

Unlike dense search, BM25 needs the full corpus in memory to score against,
so we fetch all chunks from Qdrant via scroll(), build an in-memory index,
and search that. RBAC filtering happens post-scoring here (Qdrant has no
role in BM25's ranking), unlike dense_search's native pre-filtering.

Known simplification: the index rebuilds on every call. Fine for a small
corpus; a production system would cache/persist it and rebuild only on
new data.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.chunking.indexer import COLLECTION_NAME, _get_client
from app.core.models import AccessLevel
from app.retrieval.dense_search import _allowed_levels_for


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer. BM25 just needs consistent tokens, not linguistic precision."""
    return re.findall(r"\b\w+\b", text.lower())


def _fetch_all_chunks() -> list[dict]:
    """Page through the entire Qdrant collection and return every chunk's payload + id."""
    client = _get_client()
    all_points = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break

    return [
        {
            "chunk_id": str(p.id),
            "doc_id": p.payload["doc_id"],
            "doc_title": p.payload["doc_title"],
            "access_level": p.payload["access_level"],
            "chunk_index": p.payload["chunk_index"],
            "content": p.payload["content"],
        }
        for p in all_points
    ]


def sparse_search(
    query: str,
    user_clearance: AccessLevel,
    top_k: int = 5,
) -> list[dict]:
    """
    BM25 keyword search over all chunks, filtered to the user's allowed
    access levels. Returns the same dict shape as dense_search, with a
    'score' field (BM25 score, not directly comparable to cosine similarity).
    """
    all_chunks = _fetch_all_chunks()
    if not all_chunks:
        return []

    allowed_levels = set(_allowed_levels_for(user_clearance))

    tokenized_corpus = [_tokenize(chunk["content"]) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(all_chunks, scores)
        if chunk["access_level"] in allowed_levels
    ]

    scored_chunks.sort(key=lambda c: c["score"], reverse=True)
    return scored_chunks[:top_k]