"""
Abstract interface for answer generation.

Any LLM backend (local via Ollama, or a paid API like OpenAI/Anthropic)
implements this interface. The rest of the app depends only on this
abstraction, never on a specific provider -- swapping providers later
means adding one new class, not rewriting callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models import Answer


class Generator(ABC):
    """Base class every generation provider must implement."""

    @abstractmethod
    def generate_answer(self, query: str, chunks: list[dict]) -> Answer:
        """
        Given a user query and a list of retrieved chunks (dicts with at
        least a 'content' key, as returned by hybrid_search), produce a
        grounded, cited Answer.
        """
        raise NotImplementedError