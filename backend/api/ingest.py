"""POST /ingest: upload a PDF, extract page text, return page-aware chunks.

Thin wiring layer: the shared validation/extraction pipeline lives in
``backend.api.uploads``, chunking lives in ``backend.ingestion``. Still
deliberately stateless — ``POST /index`` (Phase 2) is what persists chunks for
embeddings/FAISS.
"""
from fastapi import APIRouter, File, UploadFile

from backend.api.uploads import chunks_from_upload
from backend.ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
async def ingest_pdf(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Ingest an uploaded PDF and return its page-aware chunks as JSON.

    Query params:
        chunk_size / chunk_overlap: optional splitter overrides (validated).
    """
    filename, pages, chunks = chunks_from_upload(
        file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return _build_response(filename, pages, chunks)


def _build_response(filename: str, pages, chunks) -> dict:
    """Shape the Phase 1 JSON payload for the client."""
    page_numbers = [p.page_number for p in pages]
    return {
        "status": "success",
        "source": filename,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "page_numbers": page_numbers,
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "source": c.source,
                "text": c.text,
            }
            for c in chunks
        ],
    }
