import pytest
from pydantic import ValidationError

from app.core.models import Answer


def test_answer_valid_construction():
    answer = Answer(
        answer="Office hours are 9 AM to 6 PM.",
        citations=[0],
        confidence_score=0.92,
    )

    assert answer.answer == "Office hours are 9 AM to 6 PM."
    assert answer.citations == [0]
    assert answer.confidence_score == 0.92


def test_answer_confidence_score_must_be_between_zero_and_one():
    """A confidence_score outside [0, 1] must fail validation -- it's meant to be a probability-like signal."""
    with pytest.raises(ValidationError):
        Answer(answer="Some answer", citations=[], confidence_score=1.5)

    with pytest.raises(ValidationError):
        Answer(answer="Some answer", citations=[], confidence_score=-0.1)


def test_answer_citations_default_to_empty_list():
    """An answer with no citations field provided should default to an empty list, not error."""
    answer = Answer(answer="Some answer", confidence_score=0.5)
    assert answer.citations == []