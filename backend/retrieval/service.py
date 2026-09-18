"""Retrieval service: Phase 1 chunks -> embeddings -> persisted FAISS index.

The only place that knows about both the embedder and the vector store. It
consumes ``backend.core.models.Chunk`` objects exactly as produced by
``backend.ingestion.chunker.chunk_pages`` — no re-parsing, no re-chunking and
no new metadata fields — so `source`, `page_number`, `chunk_index` and `text`
flow through to search results unchanged.
"""
from __future__ import annotations

from pathlib import Path

from backend.core.models import Chunk, RetrievedChunk
from backend.embeddings.embedder import Embedder, get_default_embedder
from backend.ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from backend.retrieval.vector_store import (
    FaissVectorStore,
    IndexManifest,
    IndexNotBuiltError,
)

DEFAULT_STORE_DIR = Path("data/faiss_store")
DEFAULT_TOP_K = 5


class RetrievalService:
    """Index Phase 1 chunks for semantic search, persisting them to disk."""

    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE_DIR,
        embedder: Embedder | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._store_dir = Path(store_dir)
        # Building the default embedder does not load weights (lazy), so this
        # is cheap even when the service is created at import time.
        self._embedder: Embedder = embedder if embedder is not None else get_default_embedder()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._store: FaissVectorStore | None = None

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # ------------------------------------------------------------------ index
    def add_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> IndexManifest:
        """Embed ``chunks`` and persist them, appending to the existing index.

        Args:
            chunks: Phase 1 chunks to index (must be non-empty).
            reset: Replace any existing index instead of appending.
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")

        vectors = self._embedder.embed([chunk.text for chunk in chunks])

        store = None if reset else self._existing_store()
        if store is None:
            store = FaissVectorStore.build(
                chunks,
                vectors,
                model_name=self._embedder.model_name,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        else:
            store.add(
                chunks,
                vectors,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

        store.save(self._store_dir)
        self._store = store
        return store.manifest

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        store = self._existing_store()
        if store is None:
            raise IndexNotBuiltError(
                f"No index found in '{self._store_dir}'; index a document first (POST /index)"
            )

        query_vector = self._embedder.embed([query])
        return store.search(query_vector[0], top_k)

    # ------------------------------------------------------------------ state
    def _existing_store(self) -> FaissVectorStore | None:
        """Load the persisted store once; ``None`` when nothing is indexed yet."""
        if self._store is None:
            try:
                self._store = FaissVectorStore.load(
                    self._store_dir, expect_model=self._embedder.model_name
                )
            except IndexNotBuiltError:
                return None
        return self._store
