from app.generation.ollama_generator import OllamaGenerator


def test_generate_answer_grounded_in_provided_chunk():
    """The generator should produce an answer grounded in the single relevant chunk, citing it."""
    generator = OllamaGenerator()
    chunks = [
        {"content": "Our standard office hours are 9:00 AM to 6:00 PM, Monday through Friday."},
        {"content": "The company observes 12 paid holidays per year."},
    ]

    result = generator.generate_answer("What are the office hours?", chunks)

    assert "9" in result.answer  # should reference the actual hours
    assert 0 in result.citations
    assert 0.0 <= result.confidence_score <= 1.0


def test_generate_answer_with_no_chunks_returns_low_confidence():
    """With no retrieved chunks at all, the generator should not fabricate an answer."""
    generator = OllamaGenerator()

    result = generator.generate_answer("What are the office hours?", [])

    assert result.confidence_score == 0.0
    assert result.citations == []


def test_generate_answer_low_confidence_when_chunks_irrelevant():
    """When chunks don't actually address the question, confidence should be low."""
    generator = OllamaGenerator()
    chunks = [
        {"content": "The company observes 12 paid holidays per year."},
    ]

    result = generator.generate_answer("What is our gross profit margin?", chunks)

    assert result.confidence_score < 0.5