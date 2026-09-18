"""Cross-encoder reranking of retrieved chunks.

A cross-encoder scores each (query, chunk) pair jointly, which is far more
accurate than embedding cosine similarity but too slow to run over a whole
corpus — hence retrieving a wider candidate set from FAISS first and reranking
only that. The model is injectable so tests stay model-free.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from backend.core.models import RerankedChunk, RetrievedChunk

# MS MARCO passage-ranking cross-encoder: 6-layer MiniLM (~90MB), CPU-friendly.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_BATCH_SIZE = 16


@runtime_checkable
class Reranker(Protocol):
    """Minimal query + chunks -> reranked chunks contract."""

    @property
    def model_name(self) -> str:  # noqa: D102 - protocol property
        ...

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RerankedChunk]:
        """Return at most ``top_n`` chunks, best ``rerank_score`` first."""
        ...


class CrossEncoderReranker:
    """Sentence-Transformers cross-encoder reranker (CPU, lazily loaded).

    Args:
        model_name: Hugging Face cross-encoder id.
        device: Torch device string; CPU keeps Phase 3 dependency-light.
        batch_size: Batch size handed to ``predict``.
        model: Pre-built model (anything with a compatible ``predict``); used by
            tests to avoid loading weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "cpu",
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
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
        """Return the underlying cross-encoder, loading weights on first use."""
        if self._model is None:
            # Imported lazily so importing this module (and booting the app)
            # does not pull in torch.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RerankedChunk]:
        """Score every chunk against the query and keep the best ``top_n``.

        Ties keep the original retrieval order (the sort is stable), so results
        stay deterministic.
        """
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not chunks:
            return []

        model = self.load()
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(raw_scores, dtype=np.float64).reshape(len(chunks), -1)
        if scores.shape[1] != 1:
            raise ValueError(
                f"Reranker model must produce one score per pair, got {scores.shape[1]}"
            )

        # Enumerate to remember where each chunk came from in the FAISS ranking.
        scored = [(float(scores[index, 0]), index, chunk) for index, chunk in enumerate(chunks)]
        scored.sort(key=lambda item: -item[0])

        return [
            RerankedChunk(
                chunk=chunk,
                rerank_score=score,
                retrieval_rank=retrieval_rank,
                rerank_rank=rerank_rank,
            )
            for rerank_rank, (score, retrieval_rank, chunk) in enumerate(scored[:top_n])
        ]


_DEFAULT_RERANKER: CrossEncoderReranker | None = None


def get_default_reranker() -> CrossEncoderReranker:
    """Return the process-wide reranker so weights load at most once."""
    global _DEFAULT_RERANKER
    if _DEFAULT_RERANKER is None:
        _DEFAULT_RERANKER = CrossEncoderReranker()
    return _DEFAULT_RERANKER
