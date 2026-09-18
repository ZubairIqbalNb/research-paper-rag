"""Citations built from retrieved chunk metadata.

Citations are derived from ``RetrievedChunk`` fields, never parsed out of the
model's answer, so a citation can only ever reference a source/page that was
actually retrieved and passed as evidence.
"""
from __future__ import annotations

from backend.core.models import Citation, RerankedChunk


def build_citations(evidence: list[RerankedChunk]) -> list[Citation]:
    """One citation per evidence chunk, numbered to match its prompt marker.

    ``marker`` is ``rerank_rank + 1`` — the same number the prompt shows the
    model for that chunk.
    """
    return [
        Citation(
            marker=item.rerank_rank + 1,
            source=item.chunk.source,
            page_number=item.chunk.page_number,
            chunk_index=item.chunk.chunk_index,
            rerank_score=item.rerank_score,
        )
        for item in evidence
    ]
