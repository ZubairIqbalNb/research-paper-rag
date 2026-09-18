"""API tests for Phase 2: POST /index and POST /search."""
import pytest
from fastapi.testclient import TestClient

from backend.api.retrieval import get_retrieval_service
from backend.retrieval.service import RetrievalService
from backend.retrieval.vector_store import INDEX_FILENAME, METADATA_FILENAME
from health import app

PAGE_ONE = "Retrieval evaluation metrics measure ranking quality of search systems at length."
PAGE_TWO = "Photosynthesis converts sunlight into chemical energy inside plant chloroplasts."


@pytest.fixture
def service(tmp_store_dir, deterministic_embedder):
    """Service wired to a temp store and a model-free embedder."""
    return RetrievalService(store_dir=tmp_store_dir, embedder=deterministic_embedder)


@pytest.fixture
def client(service):
    app.dependency_overrides[get_retrieval_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _index(client, pdf_bytes_factory, name="paper.pdf", pages=(PAGE_ONE, PAGE_TWO), query=""):
    return client.post(
        f"/index{query}",
        files={"file": (name, pdf_bytes_factory(list(pages)), "application/pdf")},
    )


def _search(client, query, top_k=5):
    return client.post("/search", json={"query": query, "top_k": top_k})


def test_index_then_search_returns_page_metadata(client, pdf_bytes_factory):
    index_response = _index(client, pdf_bytes_factory)

    assert index_response.status_code == 200
    body = index_response.json()
    assert body["status"] == "success"
    assert body["source"] == "paper.pdf"
    assert body["num_pages"] == 2
    assert body["num_indexed"] == body["num_chunks"]
    assert body["total_vectors"] == body["num_chunks"]
    assert body["manifest"]["num_vectors"] == body["num_chunks"]

    search_response = _search(client, "retrieval evaluation metrics ranking quality")

    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert results
    top = results[0]
    assert set(top) == {"chunk_index", "page_number", "source", "text", "score"}
    assert top["source"] == "paper.pdf"
    assert top["page_number"] == 1
    assert "retrieval evaluation metrics" in top["text"].lower()
    assert results[0]["score"] >= results[-1]["score"]


def test_search_honours_top_k(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    assert len(_search(client, "retrieval evaluation metrics", top_k=1).json()["results"]) == 1
    # More than the index holds: clamped rather than padded with phantom rows.
    body = _search(client, "retrieval evaluation metrics", top_k=25).json()
    assert len(body["results"]) == body["num_results"] == 2
    assert body["top_k"] == 25


def test_search_before_indexing_returns_404(client):
    response = _search(client, "anything at all")

    assert response.status_code == 404
    assert "No index found" in response.json()["detail"]


def test_search_rejects_invalid_requests(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    assert _search(client, "valid query", top_k=0).status_code == 422
    assert _search(client, "   ", top_k=3).status_code == 422
    assert client.post("/search", json={"top_k": 3}).status_code == 422
    assert client.post("/search", json={"query": "ok"}).status_code == 200


def test_index_accumulates_documents(client, pdf_bytes_factory):
    first = _index(client, pdf_bytes_factory, name="one.pdf").json()
    second = _index(client, pdf_bytes_factory, name="two.pdf").json()

    assert second["total_vectors"] == first["num_chunks"] + second["num_chunks"]
    results = _search(client, "photosynthesis chloroplasts", top_k=5).json()["results"]
    assert {r["source"] for r in results} == {"one.pdf", "two.pdf"}


def test_index_reset_replaces_the_index(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    reset = _index(client, pdf_bytes_factory, name="single.pdf", pages=(PAGE_ONE,), query="?reset=true")

    assert reset.status_code == 200
    assert reset.json()["total_vectors"] == 1


def test_index_reuses_phase1_upload_validation(client):
    non_pdf = client.post("/index", files={"file": ("notes.txt", b"plain text", "text/plain")})
    corrupt = client.post(
        "/index",
        files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf " * 50, "application/pdf")},
    )

    assert non_pdf.status_code == 415
    assert corrupt.status_code == 422
    assert "Could not read PDF" in corrupt.json()["detail"]


def test_index_persists_store_files(client, pdf_bytes_factory, tmp_store_dir):
    _index(client, pdf_bytes_factory)

    assert (tmp_store_dir / INDEX_FILENAME).is_file()
    assert (tmp_store_dir / METADATA_FILENAME).is_file()


def test_phase1_endpoints_are_unchanged_and_stateless(client, pdf_bytes_factory, tmp_store_dir):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").status_code == 200

    response = client.post(
        "/ingest",
        files={"file": ("paper.pdf", pdf_bytes_factory([PAGE_ONE, PAGE_TWO]), "application/pdf")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["num_pages"] == 2
    assert body["page_numbers"] == [1, 2]
    assert all({"chunk_index", "page_number", "source", "text"} <= set(c) for c in body["chunks"])
    # /ingest must stay a preview: no index, no store writes.
    assert not tmp_store_dir.exists()
