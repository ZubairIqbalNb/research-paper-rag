"""Thin HTTP client used by the Streamlit frontend (Phase 5).

The frontend is deliberately *only* a client: all ingestion, embedding, FAISS
retrieval, reranking, grounded prompting and citation logic stays behind the
FastAPI backend. This module just speaks HTTP to it, preserving the architecture

    Streamlit -> FastAPI -> RAG service -> retrieval -> reranking -> Gemini

The backend address comes from the ``BACKEND_URL`` environment variable (default
``http://localhost:8000``), so no deployment-specific address is hardcoded. The
Gemini API key is backend-only configuration and is never referenced here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DEFAULT_BACKEND_URL = "http://localhost:8000"
# Indexing loads the embedding model and grounded answers call Gemini, so the
# default is deliberately generous; the health probe should stay snappy.
DEFAULT_TIMEOUT_SECONDS = 180.0
HEALTH_TIMEOUT_SECONDS = 5.0
_MAX_DETAIL_CHARS = 300

# Plain-language messages for the status codes the backend actually returns
# (see backend/api/*). The backend's own ``detail`` is appended when present.
_STATUS_MESSAGES = {
    415: "Unsupported file type - only PDF files are accepted.",
    422: "This file could not be read as a PDF (empty, corrupt, or image-only).",
    404: "No paper is indexed yet. Upload and index a PDF first.",
    500: "The backend could not read its search index.",
    502: "The language model could not produce an answer (upstream error).",
    503: "The backend is missing its Gemini API key configuration.",
}


class BackendError(Exception):
    """A backend call failed; ``str(exc)`` is safe to show a user."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class IndexResult:
    """The subset of ``POST /index`` output the UI presents."""

    source: str
    num_pages: int
    num_chunks: int
    total_vectors: int
    model_name: str


def backend_url() -> str:
    """Base URL of the backend, from ``BACKEND_URL`` (trailing slash trimmed)."""
    return (os.getenv("BACKEND_URL") or DEFAULT_BACKEND_URL).rstrip("/")


def _detail_text(response: httpx.Response) -> str:
    """Extract a short, user-safe message from a FastAPI error body."""
    try:
        payload = response.json()
    except ValueError:
        return ""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail.strip()[:_MAX_DETAIL_CHARS]
    if isinstance(detail, list):  # FastAPI request-validation error list
        parts = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        return "; ".join(parts)[:_MAX_DETAIL_CHARS]
    return ""


def _raise_for_error(response: httpx.Response) -> None:
    """Translate a non-2xx response into a friendly :class:`BackendError`."""
    if response.is_success:
        return
    message = _STATUS_MESSAGES.get(
        response.status_code, f"The backend returned HTTP {response.status_code}."
    )
    detail = _detail_text(response)
    if detail:
        message = f"{message} ({detail})"
    raise BackendError(message, status_code=response.status_code)


def _request(
    method: str,
    url: str,
    *,
    timeout: float,
    transport: httpx.BaseTransport | None = None,
    **kwargs,
) -> httpx.Response:
    """Issue one request, turning transport failures into a clear message.

    ``transport`` is an injection seam for tests (e.g. ``httpx.MockTransport``);
    it defaults to ``None``, which makes httpx use its real network transport.
    """
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return client.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        raise BackendError(
            f"The backend at {url} did not respond in time. It may still be "
            "loading models - try again in a moment."
        ) from exc
    except httpx.HTTPError as exc:
        raise BackendError(
            f"Could not reach the backend at {url}. Is the FastAPI server running?"
        ) from exc


def health(
    *,
    base_url: str | None = None,
    timeout: float = HEALTH_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Return ``True`` when the backend answers ``GET /health``.

    Raises:
        BackendError: The backend is unreachable or unhealthy.
    """
    root = (base_url or backend_url()).rstrip("/")
    response = _request("GET", f"{root}/health", timeout=timeout, transport=transport)
    _raise_for_error(response)
    return True


def ingest_pdf(
    filename: str,
    data: bytes,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Upload a PDF to ``POST /ingest`` and return the page/chunk preview.

    ``/ingest`` is the backend's stateless PDF -> pages -> chunks step; the UI
    uses the returned counts to confirm the upload was readable before indexing.
    """
    root = (base_url or backend_url()).rstrip("/")
    response = _request(
        "POST",
        f"{root}/ingest",
        timeout=timeout,
        transport=transport,
        files={"file": (filename, data, "application/pdf")},
    )
    _raise_for_error(response)
    return response.json()


def index_pdf(
    filename: str,
    data: bytes,
    *,
    reset: bool = True,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> IndexResult:
    """Upload a PDF to ``POST /index`` (embed + persist) and return its summary.

    ``reset=True`` keeps the intended "one active document" model: indexing a new
    paper replaces the previous FAISS index instead of appending to it.
    """
    root = (base_url or backend_url()).rstrip("/")
    response = _request(
        "POST",
        f"{root}/index",
        timeout=timeout,
        transport=transport,
        params={"reset": str(reset).lower()},
        files={"file": (filename, data, "application/pdf")},
    )
    _raise_for_error(response)
    body = response.json()
    return IndexResult(
        source=body["source"],
        num_pages=body["num_pages"],
        num_chunks=body["num_chunks"],
        total_vectors=body["total_vectors"],
        model_name=body["model_name"],
    )


def ask(
    question: str,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Ask ``POST /ask`` about the indexed paper and return the grounded payload.

    The response carries the answer, its grounding state, citations and both the
    FAISS and reranked evidence orderings. A refused question ("not enough
    evidence") is a successful 200 response with ``grounded == False`` - not an
    error - and is returned as-is.
    """
    root = (base_url or backend_url()).rstrip("/")
    response = _request(
        "POST",
        f"{root}/ask",
        timeout=timeout,
        transport=transport,
        json={"query": question},
    )
    _raise_for_error(response)
    return response.json()
