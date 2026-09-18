"""Shared data models for the ingestion pipeline.

Kept free of I/O so both the parser and the chunker (and later phases:
embeddings, retrieval) can depend on them without import cycles.
"""
from dataclasses import dataclass


@dataclass
class PageDocument:
    """Text extracted from a single PDF page.

    ``page_number`` is 1-based to match how humans cite pages.
    """

    text: str
    page_number: int  # 1-based
    source: str


@dataclass
class Chunk:
    """A retrievable unit of text with citation metadata."""

    text: str
    chunk_index: int  # ordinal across the whole source document
    page_number: int  # 1-based page this chunk's text came from
    source: str


@dataclass
class RetrievedChunk:
    """A search hit: a Chunk's citation metadata plus its similarity score.

    Added in Phase 2. Carries the same four metadata fields as ``Chunk`` so a
    result can always be traced back to its source document and page, plus the
    cosine similarity score produced by the vector store.
    """

    text: str
    chunk_index: int
    page_number: int  # 1-based
    source: str
    score: float  # cosine similarity in [-1, 1]
