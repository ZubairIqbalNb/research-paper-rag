"""Test doubles for Phase 2/3, kept model-free so the suite stays fast and offline.

The retrieval, reranking and RAG orchestration logic is tested through these
doubles instead of the real models and the Gemini API, so the test suite never
downloads weights, never needs an API key and never hits the network. The real
models are covered by the ``slow``-marked tests in ``tests/test_embedder.py``
and ``tests/test_reranker.py``.
"""
from __future__ import annotations

import hashlib

import numpy as np

from backend.core.models import RerankedChunk, RetrievedChunk

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


class FakeReranker:
    """Model-free reranker: lexical-overlap scores, or explicit overrides.

    ``scores`` maps chunk text -> rerank score; anything missing is scored by
    shared-word count with the query. Calls are recorded so orchestration can be
    asserted (which query, which chunks, which ``top_n``).
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        model_name: str = "deterministic-test-reranker",
    ) -> None:
        self._scores = dict(scores or {})
        self._model_name = model_name
        self.calls: list[tuple[str, list[RetrievedChunk], int]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RerankedChunk]:
        self.calls.append((query, list(chunks), top_n))
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not chunks:
            return []

        scored = [
            (self._score(query, chunk.text), index, chunk)
            for index, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda item: -item[0])  # stable: ties keep retrieval order

        return [
            RerankedChunk(
                chunk=chunk,
                rerank_score=score,
                retrieval_rank=retrieval_rank,
                rerank_rank=rerank_rank,
            )
            for rerank_rank, (score, retrieval_rank, chunk) in enumerate(scored[:top_n])
        ]

    def _score(self, query: str, text: str) -> float:
        if text in self._scores:
            return float(self._scores[text])
        return float(len(set(query.lower().split()) & set(text.lower().split())))


class FakeGenerator:
    """Records the prompts it receives and returns scripted answers.

    ``answers`` maps a substring of the prompt to the response to return, which
    lets tests drive both the grounded and the insufficient-evidence paths.
    """

    def __init__(
        self,
        answer: str = "Grounded answer based on [1].",
        answers: dict[str, str] | None = None,
        model_name: str = "fake-gemini",
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._answers = dict(answers or {})
        self._model_name = model_name
        self._error = error
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        for needle, response in self._answers.items():
            if needle in prompt:
                return response
        return self._answer
