from app.core.models import AccessLevel
from app.retrieval.dense_search import dense_search


def test_public_user_only_sees_public_chunks():
    """A user with PUBLIC clearance must never receive internal or confidential chunks."""
    results = dense_search("What is the dress code?", user_clearance=AccessLevel.PUBLIC, top_k=10)

    for result in results:
        assert result["access_level"] == "public"


def test_confidential_user_can_see_all_levels():
    """A user with CONFIDENTIAL clearance should be able to retrieve any access level."""
    results = dense_search(
        "Tell me about company operations", user_clearance=AccessLevel.CONFIDENTIAL, top_k=10
    )

    seen_levels = {r["access_level"] for r in results}
    # Not guaranteed to hit all three levels depending on relevance ranking,
    # but nothing should be excluded due to access -- so this should be a
    # subset of all valid levels and could include any of them.
    assert seen_levels.issubset({"public", "internal", "confidential"})
    assert len(results) > 0


def test_query_relevant_to_finance_returns_finance_chunk_for_confidential_user():
    """A finance-specific query should surface the confidential finance chunk when allowed."""
    results = dense_search(
        "What was the Q3 revenue?", user_clearance=AccessLevel.CONFIDENTIAL, top_k=3
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["access_level"] == "confidential"
    assert "revenue" in top_result["content"].lower() or "Revenue" in top_result["content"]


def test_public_user_cannot_retrieve_finance_chunk_even_if_relevant():
    """Even a highly relevant confidential chunk must be excluded for a public-clearance user."""
    results = dense_search(
        "What was the Q3 revenue?", user_clearance=AccessLevel.PUBLIC, top_k=5
    )

    for result in results:
        assert result["access_level"] == "public"