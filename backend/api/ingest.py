"""POST /ingest: upload a PDF, extract page text, return page-aware chunks.

Thin wiring layer: validation lives here, extraction and chunking live in
backend.ingestion. Deliberately no storage yet — a later phase will persist
chunks for embeddings/FAISS.
"""
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from backend.ingestion.pdf_parser import extract_pages

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
# UploadFile guarantees a filename; keep a defensive fallback anyway.
_FALLBACK_FILENAME = "upload.pdf"


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
    filename = file.filename or _FALLBACK_FILENAME

    # 1. Validate file type.
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type: only PDFs are accepted")

    # 2. Read and sanity-check: empty uploads and non-PDF bytes are rejected
    # early via the %PDF- magic header (real parsing errors surface later).
    data = await file.read()
    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=422,
            detail="File is empty or not a valid PDF (missing %PDF- header)",
        )

    if chunk_size < chunk_overlap or chunk_size <= 0 or chunk_overlap < 0:
        raise HTTPException(
            status_code=422,
            detail="chunk_size must be positive and >= chunk_overlap",
        )

    # 3. Persist to a temp file for pdfplumber, then extract + chunk.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        pages = extract_pages(tmp_path, source_label=filename)
    except FileNotFoundError:  # defensive: tmp file we just wrote
        raise HTTPException(status_code=500, detail="Internal error: temp file missing")
    except ValueError as exc:
        # Unreadable PDF, or no extractable text (e.g. scanned/image-only).
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
