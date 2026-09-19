"""Focused tests for the Phase 5 frontend API client.

Offline and fast: an ``httpx.MockTransport`` stands in for the FastAPI backend,
so these tests assert the exact HTTP contract the Streamlit UI relies on
(multipart uploads, query parameters, JSON bodies) and the user-facing error
mapping - without needing a running server.
"""
import json

import httpx
import pytest

from frontend import api_client

PDF_BYTES = b"%PDF-1.4\n% fake pdf for tests\n"


def _recording(payload: dict, status_code: int = 200):
    """Return ``(transport, requests)`` that records calls and replies once."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json=payload, request=request)

    return httpx.MockTransport(handler), requests


# ------------------------------------------------------------------ backend URL
class TestBackendUrl:
    def test_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("BACKEND_URL", raising=False)

        assert api_client.backend_url() == "http://localhost:8000"

    def test_reads_backend_url_and_trims_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://api.internal:9000/")

        assert api_client.backend_url() == "http://api.internal:9000"

    def test_blank_backend_url_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "")

        assert api_client.backend_url() == api_client.DEFAULT_BACKEND_URL


# ----------------------------------------------------------------------- health
class TestHealth:
    def test_healthy_backend_returns_true(self):
        transport, requests = _recording({"status": "healthy"})

        assert api_client.health(base_url="http://test", transport=transport) is True
        assert requests[0].method == "GET"
        assert requests[0].url.path == "/health"

    def test_non_2xx_raises_backend_error(self):
        transport, _ = _recording({"detail": "boom"}, status_code=500)

        with pytest.raises(api_client.BackendError) as excinfo:
            api_client.health(base_url="http://test", transport=transport)

        assert excinfo.value.status_code == 500
        assert "search index" in str(excinfo.value)

    def test_unreachable_backend_is_reported_clearly(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(api_client.BackendError, match="FastAPI server running"):
            api_client.health(base_url="http://test", transport=httpx.MockTransport(handler))

    def test_timeout_is_reported_clearly(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(api_client.BackendError, match="did not respond in time"):
            api_client.health(base_url="http://test", transport=httpx.MockTransport(handler))


# ----------------------------------------------------------------------- ingest
class TestIngest:
    def test_posts_the_file_as_multipart_and_returns_the_preview(self):
        transport, requests = _recording(
            {"status": "success", "source": "paper.pdf", "num_pages": 2, "num_chunks": 3}
        )

        preview = api_client.ingest_pdf(
            "paper.pdf", PDF_BYTES, base_url="http://test", transport=transport
        )

        assert preview["num_pages"] == 2
        request = requests[0]
        assert request.method == "POST"
        assert request.url.path == "/ingest"
        assert request.headers["content-type"].startswith("multipart/form-data")
        assert b'name="file"' in request.content
        assert b"paper.pdf" in request.content
        assert PDF_BYTES in request.content

    def test_non_pdf_upload_reports_the_415_message(self):
        transport, _ = _recording(
            {"detail": "Unsupported file type: only PDFs are accepted"}, status_code=415
        )

        with pytest.raises(api_client.BackendError, match="only PDF files are accepted"):
            api_client.ingest_pdf(
                "notes.txt", PDF_BYTES, base_url="http://test", transport=transport
            )


# ------------------------------------------------------------------------ index
class TestIndex:
    def test_posts_multipart_with_reset_and_returns_an_index_result(self):
        transport, requests = _recording(
            {
                "status": "success",
                "source": "paper.pdf",
                "num_pages": 9,
                "num_chunks": 40,
                "num_indexed": 40,
                "total_vectors": 40,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "manifest": {},
            }
        )

        result = api_client.index_pdf(
            "paper.pdf", PDF_BYTES, base_url="http://test", transport=transport
        )

        assert result.source == "paper.pdf"
        assert result.num_pages == 9
        assert result.num_chunks == 40
        assert result.total_vectors == 40
        assert result.model_name.endswith("all-MiniLM-L6-v2")
        request = requests[0]
        assert request.url.path == "/index"
        assert dict(request.url.params) == {"reset": "true"}
        assert request.headers["content-type"].startswith("multipart/form-data")

    def test_reset_false_is_sent_as_false(self):
        transport, requests = _recording(
            {
                "status": "success",
                "source": "paper.pdf",
                "num_pages": 1,
                "num_chunks": 1,
                "num_indexed": 1,
                "total_vectors": 1,
                "model_name": "m",
                "manifest": {},
            }
        )

        api_client.index_pdf(
            "paper.pdf", PDF_BYTES, reset=False, base_url="http://test", transport=transport
        )

        assert dict(requests[0].url.params) == {"reset": "false"}

    def test_unreadable_pdf_reports_the_backend_detail(self):
        transport, _ = _recording(
            {"detail": "File is empty or not a valid PDF (missing %PDF- header)"},
            status_code=422,
        )

        with pytest.raises(api_client.BackendError, match="could not be read as a PDF"):
            api_client.index_pdf("paper.pdf", PDF_BYTES, base_url="http://test", transport=transport)


# -------------------------------------------------------------------------- ask
class TestAsk:
    def _answer(self, *, grounded=True):
        return {
            "query": "what is the problem?",
            "answer": "A grounded answer [1]." if grounded else "Not enough evidence.",
            "grounded": grounded,
            "llm_model": "gemini-3.8-flash",
            "num_candidates": 1,
            "num_evidence": 0 if not grounded else 1,
            "citations": (
                [
                    {
                        "marker": 1,
                        "source": "paper.pdf",
                        "page_number": 2,
                        "chunk_index": 5,
                        "rerank_score": 3.5,
                    }
                ]
                if grounded
                else []
            ),
            "retrieved": [],
            "reranked": [],
        }

    def test_sends_the_query_as_json_and_returns_the_payload(self):
        transport, requests = _recording(self._answer())

        answer = api_client.ask(
            "what is the problem?", base_url="http://test", transport=transport
        )

        assert answer["grounded"] is True
        assert answer["citations"][0]["page_number"] == 2
        request = requests[0]
        assert request.method == "POST"
        assert request.url.path == "/ask"
        assert json.loads(request.content) == {"query": "what is the problem?"}

    def test_an_abstention_is_a_success_not_an_error(self):
        transport, _ = _recording(self._answer(grounded=False))

        answer = api_client.ask("unanswerable?", base_url="http://test", transport=transport)

        assert answer["grounded"] is False
        assert answer["citations"] == []

    def test_index_not_built_returns_the_404_message(self):
        transport, _ = _recording({"detail": "No index found"}, status_code=404)

        with pytest.raises(api_client.BackendError, match="No paper is indexed yet"):
            api_client.ask("q", base_url="http://test", transport=transport)

    def test_missing_gemini_key_returns_the_503_message(self):
        transport, _ = _recording(
            {"detail": "GEMINI_API_KEY is not configured; set it in the environment"},
            status_code=503,
        )

        with pytest.raises(api_client.BackendError, match="missing its Gemini API key"):
            api_client.ask("q", base_url="http://test", transport=transport)

    def test_llm_failure_appends_the_backend_detail(self):
        transport, _ = _recording(
            {"detail": "Gemini request failed: upstream down"}, status_code=502
        )

        with pytest.raises(api_client.BackendError, match="upstream down"):
            api_client.ask("q", base_url="http://test", transport=transport)


# ------------------------------------------------------------------- error text
class TestErrorMapping:
    def test_validation_error_list_is_flattened(self):
        transport, _ = _recording(
            {"detail": [{"msg": "Field required"}, {"msg": "Input should be a string"}]},
            status_code=422,
        )

        with pytest.raises(api_client.BackendError) as excinfo:
            api_client.ask("q", base_url="http://test", transport=transport)

        assert "Field required" in str(excinfo.value)
        assert "Input should be a string" in str(excinfo.value)

    def test_long_backend_detail_is_truncated(self):
        transport, _ = _recording({"detail": "x" * 900}, status_code=422)

        with pytest.raises(api_client.BackendError) as excinfo:
            api_client.ask("q", base_url="http://test", transport=transport)

        assert "x" * 900 not in str(excinfo.value)
        assert len(str(excinfo.value)) < 500

    def test_non_json_error_body_still_yields_a_useful_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="<html>oops</html>", request=request)

        with pytest.raises(api_client.BackendError, match="could not read its search index"):
            api_client.health(base_url="http://test", transport=httpx.MockTransport(handler))
