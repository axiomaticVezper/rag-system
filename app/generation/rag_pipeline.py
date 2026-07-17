"""
The full RAG loop: retrieve relevant chunks (RBAC-filtered, hybrid search
with reranking), then generate a grounded, cited answer from them.

This is the single entry point that ties together every prior phase.
"""

from __future__ import annotations

from app.core.models import AccessLevel, Answer
from app.generation.base import Generator
from app.generation.ollama_generator import OllamaGenerator
from app.retrieval.fusion import hybrid_search

# Default provider. Swapping to a paid API later means adding a new
# Generator subclass and changing this one line -- nothing else changes.
_default_generator: Generator = OllamaGenerator()


def answer_question(
    query: str,
    user_clearance: AccessLevel,
    top_k: int = 5,
    generator: Generator | None = None,
) -> Answer:
    """
    Run the full RAG pipeline: hybrid retrieval (RBAC-filtered) followed
    by grounded answer generation.
    """
    active_generator = generator or _default_generator

    chunks = hybrid_search(query, user_clearance, top_k=top_k)
    answer = active_generator.generate_answer(query, chunks)

    return answer