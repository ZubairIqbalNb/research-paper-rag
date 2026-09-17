"""API tests for POST /ingest using the FastAPI TestClient."""
import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from health import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoints_preserved(client):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").status_code == 200


def test_ingest_returns_page_aware_chunks(client, pdf_bytes_factory):
    content = pdf_bytes_factory([
        "First page discusses transformers and attention mechanisms at length.",
        "Second page discusses evaluation metrics for retrieval systems.",
    ])

    resp = client.post(
        "/ingest",
        files={"file": ("paper.pdf", content, "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["source"] == "paper.pdf"
    assert body["num_pages"] == 2
    assert body["num_chunks"] == len(body["chunks"]) >= 2
    assert body["page_numbers"] == [1, 2]
    # Metadata present on every chunk; page numbers 1-based.
    for chunk in body["chunks"]:
        assert {"chunk_index", "page_number", "source", "text"} <= set(chunk)
        assert chunk["source"] == "paper.pdf"
        assert chunk["page_number"] >= 1
    # First chunk carries page-1 text, some chunk carries page-2 text.
    assert any("transformers" in c["text"] for c in body["chunks"])
    assert any("evaluation metrics" in c["text"] for c in body["chunks"])
    assert [c["chunk_index"] for c in body["chunks"]] == list(range(body["num_chunks"]))


def test_ingest_rejects_non_pdf(client):
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"just text, not a pdf", "text/plain")},
    )
    assert resp.status_code == 415


def test_ingest_rejects_empty_upload(client):
    resp = client.post(
        "/ingest",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 422


def test_ingest_rejects_corrupt_pdf(client):
    resp = client.post(
        "/ingest",
        files={"file": ("corrupt.pdf", b"%PDF-1.4 this is broken content " * 100, "application/pdf")},
    )
    assert resp.status_code == 422
    assert "Could not read PDF" in resp.json()["detail"]


def test_ingest_rejects_pdf_with_no_text(client):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)

    resp = client.post(
        "/ingest",
        files={"file": ("blank.pdf", buf.getvalue(), "application/pdf")},
    )
    assert resp.status_code == 422
    assert "No extractable text" in resp.json()["detail"]


def test_ingest_validates_chunk_params(client, pdf_bytes_factory):
    content = pdf_bytes_factory(["Some perfectly fine page text."])

    resp = client.post(
        "/ingest?chunk_size=100&chunk_overlap=500",
        files={"file": ("paper.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 422
    assert "chunk_overlap" in resp.json()["detail"]


def test_ingest_custom_chunk_params_are_applied(client, pdf_bytes_factory):
    content = pdf_bytes_factory(["word " * 1000])  # ~5k chars

    resp = client.post(
        "/ingest?chunk_size=500&chunk_overlap=100",
        files={"file": ("paper.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 200
    assert all(len(c["text"]) <= 500 for c in resp.json()["chunks"])
