"""Embedder tests.

Fast tests inject a stub model, so they run without torch/weights. The real
Sentence Transformer is covered by ``@pytest.mark.slow`` integration tests
(run with ``pytest -m slow``; network needed only for the first download).
"""
import numpy as np
import pytest
from fakes import DeterministicEmbedder

from backend.embeddings.embedder import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbedder,
    get_default_embedder,
)


class _StubModel:
    """Duck-typed stand-in for SentenceTransformer that records its call.

    Returns deliberately *unnormalized* float64 rows so the test proves the
    wrapper normalizes and casts rather than trusting the model.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.arange(
            1, len(texts) * self.dimension + 1, dtype=np.float64
        ).reshape(len(texts), self.dimension)


def test_embed_returns_normalized_float32_matrix():
    embedder = SentenceTransformerEmbedder(model_name="stub", model=_StubModel(8))

    vectors = embedder.embed(["alpha text", "beta text"])

    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    assert vectors.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_embed_requests_normalized_cpu_encoding_in_batches():
    model = _StubModel(4)
    embedder = SentenceTransformerEmbedder(
        model_name="stub", model=model, batch_size=16, device="cpu"
    )

    embedder.embed(["only one text"])

    texts, kwargs = model.calls[0]
    assert texts == ["only one text"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["show_progress_bar"] is False
    assert kwargs["batch_size"] == 16
    assert kwargs["device"] == "cpu"


def test_embed_empty_input_returns_empty_matrix():
    embedder = SentenceTransformerEmbedder(model_name="stub", model=_StubModel(4))

    vectors = embedder.embed([])

    assert vectors.shape[0] == 0


def test_model_is_loaded_lazily():
    # Constructing an embedder must not touch torch/weights: the API imports
    # this module at startup, and /search works without an index.
    embedder = SentenceTransformerEmbedder()

    assert embedder.is_loaded is False
    assert get_default_embedder() is get_default_embedder()
    assert get_default_embedder().is_loaded is False


def test_stub_model_dimension_is_reported_after_load():
    embedder = SentenceTransformerEmbedder(model_name="stub", model=_StubModel(6))

    assert embedder.is_loaded is True
    assert embedder.embed(["x"]).shape == (1, 6)


def test_deterministic_embedder_prefers_shared_vocabulary():
    """Sanity-check the test double used by all retrieval tests."""
    embedder = DeterministicEmbedder()

    vectors = embedder.embed(
        ["retrieval evaluation metrics", "retrieval evaluation metrics", "photosynthesis plants"]
    )

    same = float(vectors[0] @ vectors[1])
    other = float(vectors[0] @ vectors[2])
    assert same > other
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_real_model_downloads_encodes_deterministically_and_semantically():
    pytest.importorskip("sentence_transformers")
    embedder = SentenceTransformerEmbedder(model_name=DEFAULT_MODEL_NAME, device="cpu")

    assert embedder.dimension == 384

    first = embedder.embed(["retrieval evaluation metrics for search systems"])
    second = embedder.embed(["retrieval evaluation metrics for search systems"])

    assert first.shape == (1, 384)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)  # deterministic
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)

    similar = embedder.embed(["measuring retrieval quality with ranking metrics"])
    unrelated = embedder.embed(["photosynthesis converts sunlight into sugar"])

    assert float(similar[0] @ first[0]) > float(unrelated[0] @ first[0])
