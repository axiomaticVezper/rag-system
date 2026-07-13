from app.core.models import AccessLevel
from app.retrieval.sparse_search import sparse_search


def test_exact_keyword_match_ranks_highest():
    """A query containing an exact distinctive phrase should surface the matching chunk first."""
    results = sparse_search(
        "gross margin 62 percent", user_clearance=AccessLevel.CONFIDENTIAL, top_k=3
    )

    assert len(results) > 0
    assert "margin" in results[0]["content"].lower()


def test_public_user_only_sees_public_chunks_in_sparse_search():
    """RBAC filtering must apply to sparse search results too, same as dense search."""
    results = sparse_search(
        "budget revenue finance", user_clearance=AccessLevel.PUBLIC, top_k=10
    )

    for result in results:
        assert result["access_level"] == "public"


def test_sparse_search_returns_empty_for_no_matches_gracefully():
    """A query with no meaningful term overlap should not crash, just return low/zero scores."""
    results = sparse_search(
        "xyzabc nonexistent gibberish term", user_clearance=AccessLevel.CONFIDENTIAL, top_k=5
    )

    # Should not raise; may return chunks with score 0, which is fine.
    assert isinstance(results, list)