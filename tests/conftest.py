"""Shared pytest fixtures for Phase 1 (ingestion) and Phase 2 (retrieval) tests."""
import io
import textwrap
from pathlib import Path

import pytest
from pypdf import PdfWriter

import backend.embeddings.embedder as embedder_module
import backend.reranking.reranker as reranker_module
from backend.core.models import Chunk
from backend.rag.service import RagService
from backend.retrieval.service import RetrievalService
from fakes import DeterministicEmbedder, FakeGenerator, FakeReranker


def _add_page(writer: PdfWriter, text: str) -> None:
    """Append one text page to a PDF in memory."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    text_obj = c.beginText(50, 800)
    for line in textwrap.wrap(text, width=80):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buf.seek(0)
    writer.append(buf)


@pytest.fixture
def make_pdf(tmp_path):
    """Factory: write a multi-page text PDF to tmp_path and return its Path."""
    def _make(name: str, page_texts: list[str]) -> Path:
        writer = PdfWriter()
        for text in page_texts:
            _add_page(writer, text)
        pdf_path = tmp_path / name
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path
    return _make


def _pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a multi-page text PDF fully in memory (for API uploads)."""
    writer = PdfWriter()
    for text in page_texts:
        _add_page(writer, text)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def pdf_bytes_factory():
    """Expose the in-memory PDF builder to tests (avoids import tricks)."""
    return _pdf_bytes


@pytest.fixture
def deterministic_embedder() -> DeterministicEmbedder:
    """Model-free embedder: Phase 2 tests never load (or download) weights."""
    return DeterministicEmbedder()


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Path:
    """Isolated FAISS store directory for a single test."""
    return tmp_path / "faiss_store"


@pytest.fixture
def retrieval_service(tmp_store_dir: Path, deterministic_embedder) -> RetrievalService:
    """Phase 2 retriever wired to a temp store and a model-free embedder."""
    return RetrievalService(store_dir=tmp_store_dir, embedder=deterministic_embedder)


@pytest.fixture
def fake_reranker() -> FakeReranker:
    """Model-free reranker double (records calls)."""
    return FakeReranker()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    """Model-free answer generator double (records prompts)."""
    return FakeGenerator()


@pytest.fixture
def rag_service(retrieval_service, fake_reranker, fake_generator) -> RagService:
    """RAG service with all three collaborators faked: no weights, no API key."""
    return RagService(
        retrieval=retrieval_service, reranker=fake_reranker, generator=fake_generator
    )


@pytest.fixture(autouse=True)
def _reset_default_model_singletons(monkeypatch):
    """Isolate the process-wide default embedder/reranker caches per test.

    ``get_default_embedder()`` / ``get_default_reranker()`` cache a single
    instance per module so weights load at most once per process (intended
    production behavior). The opt-in live smoke test loads the real weights on
    those shared instances; without a reset that loaded state leaks into the
    ``is_loaded is False`` lazy-loading assertions, making the suite
    order-dependent. Snapshot-and-restore around every test keeps each test
    starting from a pristine cache without touching production code.
    """
    monkeypatch.setattr(embedder_module, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(reranker_module, "_DEFAULT_RERANKER", None)


@pytest.fixture
def make_chunks():
    """Factory: build Phase 1 ``Chunk`` objects from (text, source, page) specs.

    ``chunk_index`` is assigned as a global ordinal, matching Phase 1 semantics.
    """
    def _make(specs: list[tuple[str, str, int]]) -> list[Chunk]:
        return [
            Chunk(text=text, chunk_index=index, page_number=page_number, source=source)
            for index, (text, source, page_number) in enumerate(specs)
        ]
    return _make
