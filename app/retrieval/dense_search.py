"""
Dense (vector) search over indexed chunks, with RBAC access-level filtering.

Access levels are treated as cumulative: a user's clearance grants access
to their own level and everything below it (internal sees public+internal,
confidential sees everything).
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.chunking.embedder import embed_query
from app.chunking.indexer import COLLECTION_NAME, _get_client
from app.core.models import AccessLevel

# Defines which access_level values a given clearance can see.
# Cumulative: higher clearance includes all levels below it.
_ACCESS_HIERARCHY: dict[AccessLevel, list[str]] = {
    AccessLevel.PUBLIC: [AccessLevel.PUBLIC.value],
    AccessLevel.INTERNAL: [AccessLevel.PUBLIC.value, AccessLevel.INTERNAL.value],
    AccessLevel.CONFIDENTIAL: [
        AccessLevel.PUBLIC.value,
        AccessLevel.INTERNAL.value,
        AccessLevel.CONFIDENTIAL.value,
    ],
}


def _allowed_levels_for(user_clearance: AccessLevel) -> list[str]:
    return _ACCESS_HIERARCHY[user_clearance]


def dense_search(
    query: str,
    user_clearance: AccessLevel,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the query and search Qdrant for the top_k nearest chunks,
    restricted to chunks the user's clearance is allowed to see.

    Returns a list of dicts with: chunk_id, score, doc_id, doc_title,
    access_level, chunk_index, content.
    """
    query_vector = embed_query(query)
    allowed_levels = _allowed_levels_for(user_clearance)

    access_filter = Filter(
        must=[FieldCondition(key="access_level", match=MatchAny(any=allowed_levels))]
    )

    client = _get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=access_filter,
        limit=top_k,
    )

    return [
        {
            "chunk_id": str(point.id),
            "score": point.score,
            "doc_id": point.payload["doc_id"],
            "doc_title": point.payload["doc_title"],
            "access_level": point.payload["access_level"],
            "chunk_index": point.payload["chunk_index"],
            "content": point.payload["content"],
        }
        for point in results.points
    ]