"""
Interactive RAG demo. Type a question, get a grounded, cited answer.

Run with: python scripts/ask.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.models import AccessLevel
from app.generation.rag_pipeline import answer_question
from app.retrieval.fusion import hybrid_search

_ACCESS_MAP = {
    "1": AccessLevel.PUBLIC,
    "2": AccessLevel.INTERNAL,
    "3": AccessLevel.CONFIDENTIAL,
}


def main():
    print("\n=== RAG System Demo ===")
    print("Access levels: [1] public  [2] internal  [3] confidential")
    level_input = input("Select your clearance level (1/2/3): ").strip()
    clearance = _ACCESS_MAP.get(level_input, AccessLevel.PUBLIC)
    print(f"Clearance set to: {clearance.value}\n")

    while True:
        question = input("Ask a question (or 'quit' to exit):\n> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nRetrieving relevant chunks...")
        chunks = hybrid_search(question, clearance, top_k=3)

        if not chunks:
            print("No relevant chunks found for your clearance level.")
            continue

        print(f"Found {len(chunks)} chunk(s). Generating answer...\n")
        answer = answer_question(question, clearance, top_k=3)

        print(f"Answer: {answer.answer}")
        print(f"Citations (chunk indices): {answer.citations}")
        print(f"Confidence: {answer.confidence_score:.2f}")

        print("\nSource chunks used:")
        for i, chunk in enumerate(chunks):
            print(f"  [{i}] {chunk['doc_title']} ({chunk['access_level']}) — {chunk['content'][:80]}...")

        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()