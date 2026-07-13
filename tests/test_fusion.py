from app.core.models import AccessLevel
from app.retrieval.fusion import hybrid_search, reciprocal_rank_fusion


def test_rrf_ranks_chunk_appearing_in_both_lists_highest():
    """A chunk ranked well in both dense and sparse lists should out-rank one appearing in only one."""
    dense_results = [
        {"chunk_id": "A", "content": "chunk A"},
        {"chunk_id": "B", "content": "chunk B"},
    ]
    sparse_results = [
        {"chunk_id": "A", "content": "chunk A"},
        {"chunk_id": "C", "content": "chunk C"},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    assert fused[0]["chunk_id"] == "A"  # appears in both, rank 1 in each


def test_hybrid_search_respects_rbac():
    """Fused hybrid search must still exclude chunks above the user's clearance."""
    results = hybrid_search("company finances", user_clearance=AccessLevel.PUBLIC, top_k=10)

    for result in results:
        assert result["access_level"] == "public"


def test_hybrid_search_returns_results_for_confidential_user():
    """A confidential-clearance user searching a finance term should get the finance chunk."""
    results = hybrid_search("Q3 revenue and profit margin", user_clearance=AccessLevel.CONFIDENTIAL, top_k=5)

    assert len(results) > 0
    assert any(r["access_level"] == "confidential" for r in results)