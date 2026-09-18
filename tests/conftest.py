"""Shared pytest fixtures for Phase 1 (ingestion) and Phase 2 (retrieval) tests."""
import io
import textwrap
from pathlib import Path

import pytest
from pypdf import PdfWriter

from backend.core.models import Chunk
from fakes import DeterministicEmbedder


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
