from app.core.models import AccessLevel
from app.generation.rag_pipeline import answer_question


def test_public_user_gets_grounded_answer_for_public_question():
    """A public-clearance user asking a public-appropriate question should get a real, cited answer."""
    result = answer_question("What are the office hours?", user_clearance=AccessLevel.PUBLIC)

    assert result.confidence_score > 0.3
    assert len(result.citations) > 0


def test_public_user_cannot_get_confidential_answer_even_via_full_pipeline():
    """
    Even through the full pipeline (not just raw retrieval), a public user
    asking about confidential info should not receive a confidently-grounded
    answer, since RBAC-filtered retrieval means no confidential chunks ever
    reach the generator.
    """
    result = answer_question(
        "What was our Q3 revenue and profit margin?", user_clearance=AccessLevel.PUBLIC
    )

    # No confidential chunk was ever retrieved for this user, so the model
    # should have nothing to ground a confident answer in.
    assert result.confidence_score < 0.5


def test_confidential_user_gets_grounded_finance_answer():
    """A confidential-clearance user asking the same question should get a real, grounded answer."""
    result = answer_question(
        "What was our Q3 revenue?", user_clearance=AccessLevel.CONFIDENTIAL
    )

    assert result.confidence_score > 0.3
    assert len(result.citations) > 0