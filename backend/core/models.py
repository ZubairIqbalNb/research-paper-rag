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


@dataclass
class RerankedChunk:
    """A retrieval hit re-scored by the cross-encoder.

    Added in Phase 3. Wraps the original ``RetrievedChunk`` by composition so
    no Phase 2 metadata or score is altered: ``chunk.score`` stays the FAISS
    cosine similarity and ``rerank_score`` holds the cross-encoder logit.
    Both rankings are recorded so retrieval vs reranked ordering can be compared.
    """

    chunk: RetrievedChunk
    rerank_score: float  # cross-encoder logit; higher = more relevant
    retrieval_rank: int  # 0-based position in the FAISS ordering
    rerank_rank: int  # 0-based position after reranking


@dataclass
class Citation:
    """A source/page citation derived from retrieved chunk metadata.

    Built from ``RetrievedChunk`` fields only — never parsed out of model text —
    so a citation cannot point at a page that was not actually retrieved.
    """

    marker: int  # the [n] label used in the prompt and answer
    source: str
    page_number: int  # 1-based
    chunk_index: int
    rerank_score: float


@dataclass
class RagResult:
    """Everything one RAG answer produced, for the API and for tests.

    ``prompt`` is kept for testing/debugging and is intentionally not exposed in
    the normal ``POST /ask`` response.
    """

    query: str
    retrieved: list[RetrievedChunk]  # FAISS ordering + cosine scores
    reranked: list[RerankedChunk]  # cross-encoder ordering + logits (the evidence used)
    answer: str
    grounded: bool
    citations: list[Citation]
    llm_model: str
    prompt: str
