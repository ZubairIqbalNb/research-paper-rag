"""POST /ask: grounded question answering over the indexed papers.

Thin wrapper over ``RagService``: retrieval, reranking, prompting and generation
all live in ``backend.rag``. The grounded prompt is kept on the result for
testing/debugging and is deliberately not exposed in the response.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.api.retrieval import get_retrieval_service
from backend.core.config import ConfigurationError, Settings, get_settings
from backend.core.models import RagResult
from backend.llm.client import LLMError
from backend.rag.service import (
    DEFAULT_RETRIEVAL_CANDIDATES,
    DEFAULT_TOP_N,
    MAX_RETRIEVAL_CANDIDATES,
    MAX_TOP_N,
    RagService,
)
from backend.retrieval.vector_store import IndexLoadError, IndexNotBuiltError

router = APIRouter(tags=["ask"])


@lru_cache(maxsize=1)
def get_ask_service() -> RagService:
    """Single process-wide RAG service, sharing Phase 2's retrieval service."""
    return RagService(retrieval=get_retrieval_service())


class AskRequest(BaseModel):
    """JSON body for POST /ask."""

    query: str
    candidates: int = Field(default=DEFAULT_RETRIEVAL_CANDIDATES, ge=1, le=MAX_RETRIEVAL_CANDIDATES)
    top_n: int = Field(default=DEFAULT_TOP_N, ge=1, le=MAX_TOP_N)

    @model_validator(mode="after")
    def _top_n_within_candidates(self) -> "AskRequest":
        if self.top_n > self.candidates:
            raise ValueError("top_n must be <= candidates")
        return self


@router.post("/ask")
async def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    service: RagService = Depends(get_ask_service),
) -> dict:
    """Answer a question from the indexed papers with page/source citations.

    Returns the answer, its grounding state, citations derived from chunk
    metadata, and both the FAISS and reranked evidence orderings.
    """
    # Fail fast and clearly when the LLM is not configured.
    if not settings.has_gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured; set it in the environment "
            "or in a gitignored .env file",
        )

    try:
        result = service.answer(
            request.query, candidates=request.candidates, top_n=request.top_n
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except IndexNotBuiltError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except IndexLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _serialize(result)


def _serialize(result: RagResult) -> dict:
    """Shape the /ask payload: answer, grounding, citations, evidence.

    ``result.prompt`` is intentionally omitted so the grounded prompt stays an
    internal testing/debugging detail.
    """
    return {
        "query": result.query,
        "answer": result.answer,
        "grounded": result.grounded,
        "llm_model": result.llm_model,
        "num_candidates": len(result.retrieved),
        "num_evidence": len(result.reranked),
        "citations": [
            {
                "marker": citation.marker,
                "source": citation.source,
                "page_number": citation.page_number,
                "chunk_index": citation.chunk_index,
                "rerank_score": citation.rerank_score,
            }
            for citation in result.citations
        ],
        "retrieved": [
            {
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "source": chunk.source,
                "text": chunk.text,
                "score": chunk.score,
                "retrieval_rank": rank,
            }
            for rank, chunk in enumerate(result.retrieved)
        ],
        "reranked": [
            {
                "chunk_index": item.chunk.chunk_index,
                "page_number": item.chunk.page_number,
                "source": item.chunk.source,
                "text": item.chunk.text,
                "score": item.chunk.score,
                "retrieval_rank": item.retrieval_rank,
                "rerank_rank": item.rerank_rank,
                "rerank_score": item.rerank_score,
            }
            for item in result.reranked
        ],
    }
