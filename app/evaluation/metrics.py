"""
Custom RAG evaluation metrics using a local LLM as the judge.

Three metrics:
- Faithfulness: does the answer only contain claims from the chunks?
- AnswerRelevancy: does the answer actually address the question?
- ContextualRecall: do the retrieved chunks contain what's needed to answer?

Each metric returns a float score 0.0-1.0. Higher is better.
The LLM judge is configurable so swapping to a paid model later is trivial.
"""

from __future__ import annotations

import json
import re

import numpy as np
import ollama

from app.chunking.embedder import embed_query

_DEFAULT_MODEL = "llama3.1:8b"

_JSON_SYSTEM = (
    "You are an evaluation assistant. "
    "Respond ONLY with a valid JSON object — no other text, no markdown."
)


def _call_judge(prompt: str, model: str = _DEFAULT_MODEL) -> dict:
    """Call the local LLM judge and parse its JSON response."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _JSON_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    raw = response["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)


class FaithfulnessMetric:
    """
    Measures whether the answer contains only claims supported by the
    retrieved chunks. Score = supported_claims / total_claims.
    A score < 0.5 suggests hallucination.
    """

    def score(self, answer: str, chunks: list[dict]) -> float:
        context = "\n".join(
            f"[{i}] {c['content']}" for i, c in enumerate(chunks)
        )
        prompt = f"""Given these context chunks:
{context}

And this answer:
"{answer}"

Identify each factual claim in the answer. For each claim, determine if it
is directly supported by the context chunks above.

Respond with JSON in this exact shape:
{{"total_claims": <int>, "supported_claims": <int>, "unsupported": [<list of unsupported claim strings>]}}"""

        result = _call_judge(prompt)
        total = int(result.get("total_claims", 1))
        supported = int(result.get("supported_claims", 0))
        return supported / total if total > 0 else 0.0


class AnswerRelevancyMetric:
    """
    Measures whether the answer actually addresses the question.
    Strategy: ask the LLM to generate questions the answer could address,
    then measure cosine similarity between those and the original question.
    High similarity = the answer is on-topic.
    """

    def score(self, question: str, answer: str) -> float:
        prompt = f"""Given this answer:
"{answer}"

Generate 3 different questions that this answer could be directly responding to.

Respond with JSON in this exact shape:
{{"questions": ["<question 1>", "<question 2>", "<question 3>"]}}"""

        result = _call_judge(prompt)
        generated_questions = result.get("questions", [])

        if not generated_questions:
            return 0.0

        original_vec = np.array(embed_query(question))
        similarities = []
        for gq in generated_questions:
            gq_vec = np.array(embed_query(gq))
            sim = float(np.dot(original_vec, gq_vec))
            similarities.append(sim)

        return float(np.mean(similarities))


class ContextualRecallMetric:
    """
    Measures whether the retrieved chunks contain the information needed
    to produce the expected answer. Score = fraction of expected answer's
    key facts present in the retrieved context.
    """

    def score(self, expected_answer: str, chunks: list[dict]) -> float:
        context = "\n".join(
            f"[{i}] {c['content']}" for i, c in enumerate(chunks)
        )
        prompt = f"""Given these retrieved context chunks:
{context}

And this expected answer:
"{expected_answer}"

What fraction of the key facts in the expected answer are present in the
context chunks? A fact is "present" if a chunk contains enough information
to support that specific claim.

Respond with JSON in this exact shape:
{{"total_facts": <int>, "facts_in_context": <int>, "score": <float 0.0 to 1.0>}}"""

        result = _call_judge(prompt)
        return float(result.get("score", 0.0))