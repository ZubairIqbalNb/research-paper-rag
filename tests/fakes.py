"""Test doubles for Phase 2, kept model-free so the suite stays fast and offline.

The retrieval logic (indexing, persistence, ranking, metadata mapping) is
tested through this embedder instead of the real Sentence Transformer, so the
test suite never downloads weights. The real model is covered by the
``slow``-marked integration tests in ``tests/test_embedder.py``.
"""
from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_FAKE_DIMENSION = 128


class DeterministicEmbedder:
    """Hashing bag-of-words embedder with unit-length rows.

    Texts sharing vocabulary get a higher cosine similarity, which is enough to
    assert ranking behaviour deterministically. ``blake2b`` (not the built-in
    ``hash()``, which is salted per process) keeps results stable across runs.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_FAKE_DIMENSION,
        model_name: str = "deterministic-test-embedder",
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                matrix[row, int.from_bytes(digest, "big") % self._dimension] += 1.0

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.ascontiguousarray(
            np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0),
            dtype=np.float32,
        )
