"""Local Sentence Transformer embeddings for Phase 2 retrieval.

Single responsibility: turn text into L2-normalized ``float32`` vectors. This
module knows nothing about FAISS, HTTP or chunking, and the underlying model is
injectable, so callers (and tests) can supply a stub instead of downloading
weights. Deliberately no LLM / API-backed embedding provider here.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# all-MiniLM-L6-v2: 384-dim, ~90MB, fast on CPU — the standard local default.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


@runtime_checkable
class Embedder(Protocol):
    """Minimal text -> vector contract shared by real and fake embedders."""

    @property
    def model_name(self) -> str:  # noqa: D102 - protocol property
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 matrix of unit-length rows."""
        ...


class SentenceTransformerEmbedder:
    """Sentence Transformer embedder, CPU-first, with a lazily loaded model.

    The model is only loaded on first use, so importing this module (and
    booting the API) stays cheap and works even when no index exists yet.

    Args:
        model_name: Hugging Face model id.
        device: Torch device string; CPU keeps Phase 2 dependency-light.
        batch_size: Batch size handed to ``encode``.
        model: Pre-built model (anything with a compatible ``encode``); used by
            tests to avoid loading weights, and by callers that manage their own.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are currently resident in memory."""
        return self._model is not None

    def load(self) -> object:
        """Return the underlying model, loading weights on first use."""
        if self._model is None:
            # Imported lazily: sentence-transformers pulls in torch, which is
            # slow to import and unnecessary for Phase 1 endpoints.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding size, resolved from the model (loads it if necessary)."""
        model = self.load()
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` as a normalized float32 matrix.

        An empty input yields an empty ``(0, 0)`` array — the caller decides
        whether that is an error (the retrieval service rejects empty input).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self.load()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self._device,
        )
        return _as_normalized_float32(vectors)


def _as_normalized_float32(vectors) -> np.ndarray:
    """Coerce model output into a contiguous float32 matrix with unit rows.

    The model is already asked to normalize; doing it again is idempotent and
    guarantees the store's cosine-similarity assumption holds even for
    injected/stub models.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Zero vectors (degenerate input) are left as zeros rather than dividing by 0.
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return np.ascontiguousarray(normalized, dtype=np.float32)


_DEFAULT_EMBEDDER: SentenceTransformerEmbedder | None = None


def get_default_embedder() -> SentenceTransformerEmbedder:
    """Return the process-wide embedder so weights load at most once."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder()
    return _DEFAULT_EMBEDDER
