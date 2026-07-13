"""
Cross-encoder reranking: re-scores a shortlist of fused candidates by
looking at the query and each chunk together in a single model pass,
which is more precise than the independent scoring dense/sparse search
use, but too slow to run over the whole corpus -- hence it only runs on
a small shortlist (already narrowed down by RRF fusion).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-base"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score `candidates` (dicts with a 'content' field) against `query`
    using a cross-encoder, returning the top_k most relevant, each with
    an updated 'score' reflecting the reranker's judgment.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = model.predict(pairs)

    reranked = [
        {**candidate, "score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked[:top_k]