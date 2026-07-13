"""
Local embedding generation using a sentence-transformers model.

Loads the model once (module-level singleton) since loading is slow but
reuse is fast. bge-family models are trained to expect a specific
instruction prefix on *queries* (not documents) for best retrieval
performance -- see QUERY_INSTRUCTION below.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: SentenceTransformer | None = None

# bge models are trained with this instruction prepended to *queries* only.
# Document/chunk text is embedded as-is, without this prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBEDDING_DIMENSION = 384


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of chunk texts (documents), returning one vector per text.
    Use this for indexing chunk content -- NOT for embedding a user query.
    """
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query, applying the bge query instruction prefix
    for best retrieval performance. Use this in Phase 3 for the user's
    question -- NOT for embedding document chunks.
    """
    model = _get_model()
    prefixed = f"{QUERY_INSTRUCTION}{query}"
    vector = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
    return vector.tolist()