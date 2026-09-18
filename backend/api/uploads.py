"""Shared "uploaded PDF -> page-aware chunks" pipeline for the API layer.

Both ``POST /ingest`` (Phase 1 preview) and ``POST /index`` (Phase 2 indexing)
accept the same upload, so validation and extraction live here once instead of
being duplicated per endpoint. Failures are mapped to ``HTTPException`` here
because that mapping is identical for both callers; response shaping stays in
the routers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.core.models import Chunk, PageDocument
from backend.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from backend.ingestion.pdf_parser import extract_pages

ALLOWED_CONTENT_TYPES = {"application/pdf"}
# UploadFile guarantees a filename; keep a defensive fallback anyway.
_FALLBACK_FILENAME = "upload.pdf"


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Reject splitter settings that would break the chunker."""
    if chunk_size < chunk_overlap or chunk_size <= 0 or chunk_overlap < 0:
        raise HTTPException(
            status_code=422,
            detail="chunk_size must be positive and >= chunk_overlap",
        )


def chunks_from_upload(
    file: UploadFile,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[str, list[PageDocument], list[Chunk]]:
    """Validate an uploaded PDF and return ``(filename, pages, chunks)``.

    Raises:
        HTTPException: 415 for non-PDF uploads, 422 for empty/corrupt/text-free
            PDFs or invalid chunk params, 500 if the temp file vanishes.
    """
    filename = file.filename or _FALLBACK_FILENAME

    # 1. Validate file type.
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type: only PDFs are accepted")

    # 2. Read and sanity-check: empty uploads and non-PDF bytes are rejected
    # early via the %PDF- magic header (real parsing errors surface later).
    data = file.file.read()
    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=422,
            detail="File is empty or not a valid PDF (missing %PDF- header)",
        )

    validate_chunk_params(chunk_size, chunk_overlap)

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
    return filename, pages, chunks
