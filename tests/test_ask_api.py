"""API tests for Phase 3 POST /ask, plus Phase 1/2 regression checks."""
import os

import pytest
from fastapi.testclient import TestClient

from backend.api.ask import get_ask_service
from backend.api.retrieval import get_retrieval_service
from backend.core.config import Settings, get_settings
from backend.llm.client import LLMError
from backend.rag.prompt import INSUFFICIENT_EVIDENCE_MARKER
from backend.rag.service import RagService
from backend.reranking.reranker import DEFAULT_RERANK_MODEL
from fakes import FakeGenerator, FakeReranker
from health import app

PAGE_ONE = "Retrieval evaluation metrics measure ranking quality of search systems at length."
PAGE_TWO = "Photosynthesis converts sunlight into chemical energy inside plant chloroplasts."

ASK_RESPONSE_KEYS = {
    "query",
    "answer",
    "grounded",
    "llm_model",
    "num_candidates",
    "num_evidence",
    "citations",
    "retrieved",
    "reranked",
}


@pytest.fixture
def client(retrieval_service, rag_service) -> TestClient:
    """TestClient with the Phase 2 retriever and the RAG service both stubbed."""
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_ask_service] = lambda: rag_service
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="test-key-not-real", gemini_model="gemini-test-model"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _index(client, pdf_bytes_factory, name="paper.pdf", pages=(PAGE_ONE, PAGE_TWO)):
    return client.post(
        "/index", files={"file": (name, pdf_bytes_factory(list(pages)), "application/pdf")}
    )


def _ask(client, **body):
    return client.post("/ask", json=body)


# ----------------------------------------------------------------- happy path
def test_ask_returns_grounded_answer_with_citations(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    response = _ask(client, query="retrieval evaluation metrics ranking quality")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == ASK_RESPONSE_KEYS
    assert body["grounded"] is True
    assert body["answer"] == "Grounded answer based on [1]."
    assert body["llm_model"] == "fake-gemini"
    assert body["citations"]
    top_citation = body["citations"][0]
    assert set(top_citation) == {
        "marker",
        "source",
        "page_number",
        "chunk_index",
        "rerank_score",
    }
    assert top_citation["marker"] == 1
    assert top_citation["source"] == "paper.pdf"
    assert top_citation["page_number"] >= 1


def test_ask_does_not_expose_the_internal_prompt(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    response = _ask(client, query="retrieval evaluation metrics ranking quality")

    assert "prompt" not in response.json()
    # No fragment of the grounded prompt leaks through any field.
    assert "using ONLY the numbered evidence" not in response.text
    assert INSUFFICIENT_EVIDENCE_MARKER not in response.text


def test_ask_exposes_both_retrieval_and_reranked_orderings(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    body = _ask(client, query="retrieval evaluation metrics ranking quality", top_n=2).json()

    assert body["num_candidates"] == 2
    assert body["num_evidence"] == 2
    assert [item["retrieval_rank"] for item in body["retrieved"]] == [0, 1]
    for item in body["retrieved"]:
        assert item["score"] is not None  # FAISS cosine score, Phase 2 semantics
        assert "rerank_score" not in item
    for rank, item in enumerate(body["reranked"]):
        assert item["rerank_rank"] == rank
        assert item["score"] is not None  # cosine score preserved
        assert item["rerank_score"] is not None  # cross-encoder logit, separate
        assert 0 <= item["retrieval_rank"] < body["num_candidates"]
        assert item["page_number"] >= 1 and item["text"]


def test_ask_honours_candidates_and_top_n(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)

    body = _ask(client, query="retrieval evaluation metrics", candidates=1, top_n=1).json()

    assert body["num_candidates"] == 1 and body["num_evidence"] == 1


# ---------------------------------------------------------------- grounding
def test_ask_insufficient_evidence_returns_200_ungrounded(
    client, pdf_bytes_factory, retrieval_service
):
    _index(client, pdf_bytes_factory)
    app.dependency_overrides[get_ask_service] = lambda: RagService(
        retrieval=retrieval_service,
        reranker=FakeReranker(),
        generator=FakeGenerator(
            answer=f"{INSUFFICIENT_EVIDENCE_MARKER}\nThe papers do not report this."
        ),
    )

    response = _ask(client, query="what is the capital of France")

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["answer"] == "The papers do not report this."
    assert body["citations"] == []
    assert body["reranked"]  # evidence is still reported for inspection


def test_ask_without_index_returns_404(client):
    response = _ask(client, query="anything at all")

    assert response.status_code == 404
    assert "No index found" in response.json()["detail"]


# -------------------------------------------------------------------- errors
def test_ask_without_api_key_returns_503(client, pdf_bytes_factory):
    _index(client, pdf_bytes_factory)
    app.dependency_overrides[get_settings] = lambda: Settings(gemini_api_key=None)

    response = _ask(client, query="retrieval evaluation metrics")

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_ask_returns_502_when_generation_fails(client, pdf_bytes_factory, retrieval_service):
    _index(client, pdf_bytes_factory)
    app.dependency_overrides[get_ask_service] = lambda: RagService(
        retrieval=retrieval_service,
        reranker=FakeReranker(),
        generator=FakeGenerator(error=LLMError("Gemini request failed: upstream down")),
    )

    response = _ask(client, query="retrieval evaluation metrics")

    assert response.status_code == 502
    assert "upstream down" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "   "},
        {"query": "valid query", "top_n": 0},
        {"query": "valid query", "top_n": 21},
        {"query": "valid query", "candidates": 0},
        {"query": "valid query", "candidates": 101},
        {"query": "valid query", "candidates": 2, "top_n": 3},
        {"top_k": 3},
    ],
)
def test_ask_rejects_invalid_requests(client, pdf_bytes_factory, body):
    _index(client, pdf_bytes_factory)

    assert _ask(client, **body).status_code == 422


def test_ask_validation_happens_before_the_llm_is_configured_check(client):
    # Missing query -> 422 (schema), not 503 (missing key).
    app.dependency_overrides[get_settings] = lambda: Settings(gemini_api_key=None)

    assert client.post("/ask", json={}).status_code == 422


# -------------------------------------------------------- Phase 1/2 regression
def test_phase1_and_phase2_endpoints_still_work(client, pdf_bytes_factory, tmp_store_dir):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").status_code == 200

    ingest = client.post(
        "/ingest",
        files={"file": ("paper.pdf", pdf_bytes_factory([PAGE_ONE, PAGE_TWO]), "application/pdf")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["page_numbers"] == [1, 2]

    indexed = _index(client, pdf_bytes_factory)
    assert indexed.status_code == 200
    assert indexed.json()["total_vectors"] == 2
    assert (tmp_store_dir / "index.faiss").is_file()

    search = client.post("/search", json={"query": "retrieval evaluation metrics", "top_k": 2})
    assert search.status_code == 200
    body = search.json()
    assert set(body) == {"query", "top_k", "num_results", "results"}
    assert set(body["results"][0]) == {"chunk_index", "page_number", "source", "text", "score"}


def test_ask_endpoint_is_documented(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) >= {"/ingest", "/index", "/search", "/ask", "/health"}


# --------------------------------------------------------------- live smoke
@pytest.mark.live
@pytest.mark.slow
def test_live_ask_end_to_end(tmp_path, pdf_bytes_factory):
    """Real embedder + real cross-encoder + real Gemini. Opt-in only."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set in the process environment")

    from backend.embeddings.embedder import get_default_embedder
    from backend.reranking.reranker import get_default_reranker
    from backend.retrieval.service import RetrievalService

    retrieval = RetrievalService(store_dir=tmp_path / "live_store", embedder=get_default_embedder())
    service = RagService(
        retrieval=retrieval, reranker=get_default_reranker(), generator=None
    )
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_ask_service] = lambda: service
    try:
        client = TestClient(app)
        indexed = client.post(
            "/index",
            files={
                "file": (
                    "live.pdf",
                    pdf_bytes_factory([PAGE_ONE, PAGE_TWO]),
                    "application/pdf",
                )
            },
        )
        assert indexed.status_code == 200

        response = client.post(
            "/ask", json={"query": "how is retrieval quality measured?", "top_n": 3}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"].strip()
        assert body["grounded"] is True
        assert body["citations"]
        assert body["citations"][0]["source"] == "live.pdf"
        assert body["llm_model"]
    finally:
        app.dependency_overrides.clear()


def test_live_marker_registered_and_default_rerank_model_is_the_cpu_minilm():
    assert DEFAULT_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
