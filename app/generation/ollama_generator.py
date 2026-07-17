"""
Ollama-backed implementation of the Generator interface, using a local
LLM (default: llama3.1:8b). Uses Ollama's JSON mode plus explicit prompt
instructions (schema + few-shot example) to maximize structured-output
reliability, and retries once with the parse error fed back to the model
if the first response doesn't validate against the Answer schema.
"""

from __future__ import annotations

import json

import ollama
from pydantic import ValidationError

from app.core.models import Answer
from app.generation.base import Generator

_SYSTEM_PROMPT = """You are a precise question-answering assistant. You will be given a user question and a numbered list of context chunks retrieved from a document database.

Rules:
- Answer ONLY using information from the provided chunks. Do not use outside knowledge.
- If the chunks do not contain enough information to answer, say so explicitly in your answer, and set confidence_score low (below 0.3).
- Cite every chunk index you actually used in your answer, in the "citations" field. Do not cite chunks you didn't use.
- confidence_score reflects how well the chunks support your answer: 1.0 means fully and directly supported, 0.0 means not supported at all.
- Respond with ONLY a single JSON object in this exact shape, no other text:
{"answer": "<your answer text>", "citations": [<chunk indices used, as integers>], "confidence_score": <float between 0 and 1>}

Example:
Question: What are the office hours?
Chunks:
[0] Our standard office hours are 9:00 AM to 6:00 PM, Monday through Friday.
[1] The company observes 12 paid holidays per year.

Correct response:
{"answer": "Office hours are 9:00 AM to 6:00 PM, Monday through Friday.", "citations": [0], "confidence_score": 0.95}
"""


class OllamaGenerator(Generator):
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def _build_user_prompt(self, query: str, chunks: list[dict]) -> str:
        numbered_chunks = "\n".join(
            f"[{i}] {chunk['content']}" for i, chunk in enumerate(chunks)
        )
        return f"Question: {query}\n\nChunks:\n{numbered_chunks}\n\nRespond with only the JSON object."

    def _call_model(self, user_prompt: str, error_feedback: str | None = None) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if error_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {error_feedback}. "
                        "Respond again with ONLY a valid JSON object matching the required shape."
                    ),
                }
            )

        response = ollama.chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        return response["message"]["content"]

    def generate_answer(self, query: str, chunks: list[dict]) -> Answer:
        if not chunks:
            return Answer(
                answer="I don't have any relevant information to answer this question.",
                citations=[],
                confidence_score=0.0,
            )

        user_prompt = self._build_user_prompt(query, chunks)
        raw_response = self._call_model(user_prompt)

        try:
            parsed = json.loads(raw_response)
            return Answer(**parsed)
        except (json.JSONDecodeError, ValidationError) as first_error:
            # Retry once, feeding the error back to the model.
            raw_response = self._call_model(user_prompt, error_feedback=str(first_error))
            parsed = json.loads(raw_response)  # let this raise if it fails again
            return Answer(**parsed)