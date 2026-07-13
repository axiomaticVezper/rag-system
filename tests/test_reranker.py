from app.core.models import AccessLevel
from app.retrieval.fusion import hybrid_search
from app.retrieval.reranker import rerank


def test_rerank_reorders_by_relevance():
    """The reranker should score a more relevant chunk above a less relevant one."""
    candidates = [
        {"chunk_id": "1", "content": "The company observes 12 paid holidays per year."},
        {"chunk_id": "2", "content": "Gross margin stands at 62 percent this quarter."},
    ]

    results = rerank("What is our profit margin?", candidates, top_k=2)

    assert results[0]["chunk_id"] == "2"


def test_rerank_empty_candidates_returns_empty():
    assert rerank("any query", [], top_k=5) == []


def test_hybrid_search_with_reranker_respects_rbac():
    """Reranking must not bypass the access-level filtering already applied upstream."""
    results = hybrid_search(
        "company finances and revenue", user_clearance=AccessLevel.PUBLIC, top_k=5, use_reranker=True
    )

    for result in results:
        assert result["access_level"] == "public"


def test_hybrid_search_reranker_can_be_disabled():
    """use_reranker=False should skip reranking and just return fused RRF results."""
    results = hybrid_search(
        "office hours", user_clearance=AccessLevel.PUBLIC, top_k=3, use_reranker=False
    )

    assert len(results) > 0