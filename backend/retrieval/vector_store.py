"""FAISS-backed vector store with a JSON metadata sidecar.

Responsibilities:
  * hold an exact inner-product FAISS index over L2-normalized embeddings
    (inner product on unit vectors == cosine similarity)
  * keep chunk metadata (source / page_number / chunk_index / text) in a
    separate JSON file, so vectors and citation metadata stay decoupled
  * persist and reload the pair, verifying they still agree

A FAISS flat index identifies vectors by *position*, not by id, so the
invariant that matters is:

    index row ``i``  <->  ``metadata["chunks"][i]``

It is established on build, extended on append, and re-verified on load.
``Chunk.chunk_index`` is a per-document ordinal (Phase 1), not a global id, so
it is never used to address a vector.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from backend.core.models import Chunk, RetrievedChunk

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1


class IndexNotBuiltError(Exception):
    """Raised when no persisted index exists yet."""


class IndexLoadError(Exception):
    """Raised when a persisted index is unreadable or internally inconsistent."""


@dataclass
class IndexManifest:
    """Self-describing header so a persisted index can be validated/rebuilt."""

    model_name: str
    embedding_dim: int
    num_vectors: int
    normalized: bool = True
    # Chunking params used to produce the stored chunks; None means the store
    # mixes documents chunked with different params.
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexManifest":
        """Build a manifest from persisted JSON, ignoring unknown keys."""
        return cls(
            model_name=str(data["model_name"]),
            embedding_dim=int(data["embedding_dim"]),
            num_vectors=int(data["num_vectors"]),
            normalized=bool(data.get("normalized", True)),
            chunk_size=data.get("chunk_size"),
            chunk_overlap=data.get("chunk_overlap"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            created_at=str(data.get("created_at", "")),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_matrix(vectors) -> np.ndarray:
    """Coerce vectors into a contiguous 2-D float32 array (FAISS requirement)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D embedding array, got shape {matrix.shape}")
    return np.ascontiguousarray(matrix, dtype=np.float32)


class FaissVectorStore:
    """An in-memory FAISS index plus the chunk metadata for each row."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: IndexManifest) -> None:
        self._index = index
        self._chunks = list(chunks)
        self._manifest = manifest

    # ------------------------------------------------------------------ state
    @property
    def size(self) -> int:
        """Number of indexed vectors (== number of metadata rows)."""
        return int(self._index.ntotal)

    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @property
    def chunks(self) -> list[Chunk]:
        """A copy of the chunk metadata, in FAISS row order."""
        return list(self._chunks)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        vectors,
        *,
        model_name: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> "FaissVectorStore":
        """Create an exact inner-product index over the given chunks."""
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list")

        matrix = _as_matrix(vectors)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Got {matrix.shape[0]} vectors for {len(chunks)} chunks; "
                "vectors and chunk metadata must line up one-to-one"
            )

        dimension = int(matrix.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)

        manifest = IndexManifest(
            model_name=model_name,
            embedding_dim=dimension,
            num_vectors=len(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=_utc_now(),
        )
        return cls(index, chunks, manifest)

    def add(
        self,
        chunks: list[Chunk],
        vectors,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IndexManifest:
        """Append chunks and their vectors, keeping row order intact."""
        if not chunks:
            raise ValueError("Cannot append an empty chunk list")

        matrix = _as_matrix(vectors)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Got {matrix.shape[0]} vectors for {len(chunks)} chunks; "
                "vectors and chunk metadata must line up one-to-one"
            )
        if matrix.shape[1] != self._manifest.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: index holds "
                f"{self._manifest.embedding_dim}-dim vectors, got {matrix.shape[1]}-dim"
            )

        self._index.add(matrix)
        self._chunks.extend(chunks)

        stored = (self._manifest.chunk_size, self._manifest.chunk_overlap)
        incoming = (chunk_size, chunk_overlap)
        # Documents chunked with different params make the stored params ambiguous.
        chunk_params = stored if incoming == stored else (None, None)

        self._manifest = replace(
            self._manifest,
            num_vectors=self.size,
            chunk_size=chunk_params[0],
            chunk_overlap=chunk_params[1],
        )
        return self._manifest

    # ----------------------------------------------------------------- search
    def search(self, query_vector, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best score first."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.size == 0:
            return []

        query = _as_matrix(query_vector)
        if query.shape[1] != self._manifest.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: index holds "
                f"{self._manifest.embedding_dim}-dim vectors, got {query.shape[1]}-dim"
            )

        # FAISS needs k <= ntotal; requesting more would yield -1 padding rows.
        k = min(top_k, self.size)
        scores, rows = self._index.search(query, k)

        results: list[RetrievedChunk] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:  # defensive: FAISS pads with -1 when short on neighbours
                continue
            chunk = self._chunks[int(row)]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    source=chunk.source,
                    score=float(score),
                )
            )
        return results

    # ------------------------------------------------------------ persistence
    def save(self, store_dir: Path) -> None:
        """Write index + metadata to ``store_dir``, swapping both atomically."""
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        index_path = store_dir / INDEX_FILENAME
        metadata_path = store_dir / METADATA_FILENAME
        tmp_index = store_dir / f".{INDEX_FILENAME}.tmp"
        tmp_metadata = store_dir / f".{METADATA_FILENAME}.tmp"

        payload = {
            "manifest": replace(self._manifest, num_vectors=self.size).to_dict(),
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "source": chunk.source,
                    "text": chunk.text,
                }
                for chunk in self._chunks
            ],
        }

        try:
            # Write both temporaries first, then swap, so a failure never leaves
            # a half-updated store behind.
            faiss.write_index(self._index, str(tmp_index))
            tmp_metadata.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_index, index_path)
            os.replace(tmp_metadata, metadata_path)
        finally:
            tmp_index.unlink(missing_ok=True)
            tmp_metadata.unlink(missing_ok=True)

    @classmethod
    def load(cls, store_dir: Path, *, expect_model: str | None = None) -> "FaissVectorStore":
        """Load a persisted store, verifying index and metadata still agree.

        Raises:
            IndexNotBuiltError: No persisted index exists in ``store_dir``.
            IndexLoadError: Files exist but are corrupt or inconsistent.
        """
        store_dir = Path(store_dir)
        index_path = store_dir / INDEX_FILENAME
        metadata_path = store_dir / METADATA_FILENAME

        if not index_path.is_file() or not metadata_path.is_file():
            raise IndexNotBuiltError(
                f"No index found in '{store_dir}'; index a document first (POST /index)"
            )

        try:
            index = faiss.read_index(str(index_path))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = IndexManifest.from_dict(payload["manifest"])
            chunks = [
                Chunk(
                    text=row["text"],
                    chunk_index=int(row["chunk_index"]),
                    page_number=int(row["page_number"]),
                    source=row["source"],
                )
                for row in payload["chunks"]
            ]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            raise IndexLoadError(f"Could not read index in '{store_dir}': {exc}") from exc

        if manifest.schema_version > SCHEMA_VERSION:
            raise IndexLoadError(
                f"Index schema version {manifest.schema_version} is newer than the "
                f"supported version {SCHEMA_VERSION}"
            )
        if len(chunks) != manifest.num_vectors:
            raise IndexLoadError(
                f"Metadata is inconsistent: {len(chunks)} chunk rows for a manifest "
                f"declaring {manifest.num_vectors} vectors"
            )
        if index.ntotal != manifest.num_vectors:
            raise IndexLoadError(
                f"Index/metadata mismatch: {index.ntotal} vectors on disk but "
                f"{manifest.num_vectors} chunk rows"
            )
        if index.d != manifest.embedding_dim:
            raise IndexLoadError(
                f"Dimension mismatch: index holds {index.d}-dim vectors but the "
                f"manifest declares {manifest.embedding_dim}"
            )
        if expect_model is not None and manifest.model_name != expect_model:
            raise IndexLoadError(
                f"Index was built with embedding model '{manifest.model_name}' but "
                f"'{expect_model}' is configured"
            )

        return cls(index, chunks, manifest)
