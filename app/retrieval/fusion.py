"""
Reciprocal Rank Fusion (RRF): combines dense and sparse search result
lists into a single ranked list, using only rank position (not raw
scores), which sidesteps the problem of dense (cosine, ~0-1) and BM25
(unbounded) scores being on incompatible scales.
"""

from __future__ import annotations
from app.retrieval.reranker import rerank
from app.core.models import AccessLevel
from app.retrieval.dense_search import dense_search
from app.retrieval.sparse_search import sparse_search

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    Fuse two ranked result lists by chunk_id using RRF scoring.

    Each chunk's fused score is 1/(k + rank) summed across whichever
    list(s) it appears in (rank is 1-indexed). Chunks appearing in both
    lists get contributions from both, naturally ranking them higher.
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_data[chunk_id] = result

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_data.setdefault(chunk_id, result)

    fused = [
        {**chunk_data[chunk_id], "score": rrf_score}
        for chunk_id, rrf_score in rrf_scores.items()
    ]
    fused.sort(key=lambda c: c["score"], reverse=True)
    return fused


def hybrid_search(
    query: str,
    user_clearance: AccessLevel,
    top_k: int = 5,
    candidate_pool_size: int = 20,
    use_reranker: bool = True,
) -> list[dict]:
    """
    Run dense and sparse search in parallel (conceptually; sequential here
    for simplicity), fuse with RRF, and return the top_k fused results.

    candidate_pool_size controls how many results each individual method
    contributes before fusion -- wider than top_k so fusion has enough
    material to work with.
    """
    dense_results = dense_search(query, user_clearance, top_k=candidate_pool_size)
    sparse_results = sparse_search(query, user_clearance, top_k=candidate_pool_size)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    if use_reranker:
        # Rerank a slightly wider slice than top_k so the cross-encoder
        # has real alternatives to compare, not just the already-final list.
        shortlist = fused[: max(top_k * 3, 10)]
        return rerank(query, shortlist, top_k=top_k)

    return fused[:top_k]