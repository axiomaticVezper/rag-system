from app.chunking.embedder import EMBEDDING_DIMENSION, embed_query, embed_texts


def test_embed_texts_returns_correct_dimension():
    """Each embedded text should produce a vector of the expected dimension."""
    vectors = embed_texts(["Office hours are 9 to 6.", "The dress code is business casual."])

    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    assert len(vectors[1]) == EMBEDDING_DIMENSION


def test_embed_query_returns_correct_dimension():
    """A single embedded query should produce one vector of the expected dimension."""
    vector = embed_query("What are the office hours?")

    assert len(vector) == EMBEDDING_DIMENSION


def test_similar_texts_produce_closer_vectors_than_dissimilar_ones():
    """
    Semantically similar sentences should have a higher cosine similarity
    than semantically unrelated ones -- proving the embeddings actually
    capture meaning, not just producing arbitrary numbers.
    """
    import numpy as np

    vectors = embed_texts([
        "Office hours are 9 AM to 6 PM.",
        "Our work day runs from 9 in the morning until 6 in the evening.",
        "The gross profit margin was 62 percent this quarter.",
    ])

    v_office, v_office_paraphrase, v_finance = [np.array(v) for v in vectors]

    sim_similar = np.dot(v_office, v_office_paraphrase)
    sim_different = np.dot(v_office, v_finance)

    assert sim_similar > sim_different