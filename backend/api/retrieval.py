"""Phase 2 endpoints: POST /index (embed + persist) and POST /search.

Semantic retrieval only — no reranking, no query routing, no LLM/RAG. Both
endpoints are thin wrappers over ``RetrievalService``; validation stays here.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api.uploads import chunks_from_upload
from backend.ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from backend.retrieval.service import DEFAULT_TOP_K, RetrievalService
from backend.retrieval.vector_store import IndexLoadError, IndexNotBuiltError

router = APIRouter(tags=["retrieval"])


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Single process-wide service: one embedder, one in-memory index."""
    return RetrievalService()


class SearchRequest(BaseModel):
    """JSON body for POST /search."""

    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1)


@router.post("/index")
async def index_pdf(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
    service: RetrievalService = Depends(get_retrieval_service),
) -> dict:
    """Embed an uploaded PDF's chunks into the persisted FAISS index.

    Uses the same PDF -> pages -> chunks pipeline as POST /ingest, so the
    indexed metadata is exactly what ``/ingest`` returns. Appends to an
    existing index unless ``reset=true``.

    Query params:
        chunk_size / chunk_overlap: optional splitter overrides (validated).
        reset: replace the existing index instead of appending.
    """
    filename, pages, chunks = chunks_from_upload(
        file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    try:
        manifest = service.add_chunks(chunks, reset=reset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except IndexLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "source": filename,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "num_indexed": len(chunks),
        "total_vectors": manifest.num_vectors,
        "model_name": manifest.model_name,
        "manifest": manifest.to_dict(),
    }


@router.post("/search")
async def search(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> dict:
    """Return the chunks most semantically similar to the query."""
    try:
        results = service.search(request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except IndexNotBuiltError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except IndexLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": request.query,
        "top_k": request.top_k,
        "num_results": len(results),
        "results": [
            {
                "chunk_index": result.chunk_index,
                "page_number": result.page_number,
                "source": result.source,
                "text": result.text,
                "score": result.score,
            }
            for result in results
        ],
    }
