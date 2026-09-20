"""Lightweight UI tests for the Streamlit app.

Uses Streamlit's own ``AppTest`` runner (no extra framework, no browser, no
network): the backend URL points at a closed port so the app's health probe
fails fast and the UI must degrade gracefully rather than crash.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client
from frontend import app as frontend_app

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
# A closed port: connecting fails instantly, keeping the tests fast and offline.
UNREACHABLE_BACKEND = "http://127.0.0.1:9"

INDEXED = {
    "source": "resnet.pdf",
    "num_pages": 12,
    "num_chunks": 70,
    "total_vectors": 70,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
}


def _grounded_answer() -> dict:
    """A realistic POST /ask payload for an answerable question."""
    return {
        "query": "What problem does depth cause?",
        "answer": "Deeper plain networks suffer degradation. [1]",
        "grounded": True,
        "llm_model": "gemini-3.8-flash",
        "num_candidates": 1,
        "num_evidence": 1,
        "citations": [
            {
                "marker": 1,
                "source": "resnet.pdf",
                "page_number": 2,
                "chunk_index": 5,
                "rerank_score": 3.5,
            }
        ],
        "retrieved": [
            {
                "chunk_index": 5,
                "page_number": 2,
                "source": "resnet.pdf",
                "text": "degradation of deeper plain networks",
                "score": 0.71,
                "retrieval_rank": 0,
            }
        ],
        "reranked": [
            {
                "chunk_index": 5,
                "page_number": 2,
                "source": "resnet.pdf",
                "text": "degradation of deeper plain networks",
                "score": 0.71,
                "retrieval_rank": 0,
                "rerank_rank": 0,
                "rerank_score": 3.5,
            }
        ],
    }


def _abstained_answer() -> dict:
    """A realistic POST /ask payload for an unanswerable question."""
    return {
        "query": "What was the inference latency?",
        "answer": "The indexed papers do not contain enough evidence to answer this question.",
        "grounded": False,
        "llm_model": "gemini-3.8-flash",
        "num_candidates": 1,
        "num_evidence": 1,
        "citations": [],
        "retrieved": [
            {
                "chunk_index": 5,
                "page_number": 2,
                "source": "resnet.pdf",
                "text": "some unrelated passage",
                "score": 0.31,
                "retrieval_rank": 0,
            }
        ],
        "reranked": [
            {
                "chunk_index": 5,
                "page_number": 2,
                "source": "resnet.pdf",
                "text": "some unrelated passage",
                "score": 0.31,
                "retrieval_rank": 0,
                "rerank_rank": 0,
                "rerank_score": -4.2,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _point_at_unreachable_backend(monkeypatch):
    """Keep the app's health probe offline and instantaneous."""
    monkeypatch.setenv("BACKEND_URL", UNREACHABLE_BACKEND)


def _run(**state):
    """Run the app once with the given session state pre-seeded."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


# ------------------------------------------------------------------- baseline
class TestAppRenders:
    def test_renders_without_exceptions_and_shows_both_steps(self):
        at = _run()

        assert not at.exception
        assert [title.value for title in at.title] == ["Research Paper Digest"]
        assert [header.value for header in at.header] == [
            "1. Upload a paper",
            "2. Ask a question",
        ]
        assert len(at.get("file_uploader")) == 1

    def test_uploader_accepts_pdfs_only(self):
        at = _run()

        uploader = at.get("file_uploader")[0]
        assert list(uploader.proto.type) == [".pdf"]

    def test_index_button_appears_only_after_a_file_is_selected(self):
        at = _run()

        assert not any(button.label == "Ingest and index this paper" for button in at.button)

    def test_unreachable_backend_is_communicated_not_crashed(self):
        at = _run()

        assert not at.exception
        assert any("not reachable" in error.value for error in at.sidebar.error)

    def test_ask_is_unavailable_until_a_paper_is_indexed(self):
        at = _run()

        assert not at.exception
        assert at.text_input == []
        assert any("Upload and index a paper first" in info.value for info in at.info)


# --------------------------------------------------------------------- indexed
class TestIndexedState:
    def test_indexed_paper_shows_status_and_unlocks_the_question_box(self):
        at = _run(indexed=INDEXED, backend_ok=True)

        assert not at.exception
        status = " ".join(info.value for info in at.info)
        assert "resnet.pdf" in status and "12 pages" in status
        assert [box.label for box in at.text_input] == ["Your question about the paper"]

    def test_clear_paper_resets_the_session(self):
        at = _run(indexed=INDEXED, backend_ok=True)
        clear = next(button for button in at.button if button.label == "Clear paper")

        clear.click().run()

        assert at.session_state["indexed"] is None


# ---------------------------------------------------------------------- answer
class TestAnswerRendering:
    def test_grounded_answer_shows_sources_and_evidence(self):
        at = _run(
            indexed=INDEXED,
            backend_ok=True,
            answer=_grounded_answer(),
            asked_question="What problem does depth cause?",
        )

        assert not at.exception
        assert any("Grounded in the indexed paper." in s.value for s in at.success)
        markdown = " ".join(block.value for block in at.markdown)
        assert "Page **2**" in markdown  # citation
        assert "degradation of deeper plain networks" in markdown  # evidence text
        assert any(expander.label == "Retrieved Evidence" for expander in at.expander)

    def test_abstention_is_a_warning_not_an_error(self):
        at = _run(
            indexed=INDEXED,
            backend_ok=True,
            answer=_abstained_answer(),
            asked_question="What was the inference latency?",
        )

        assert not at.exception
        assert any("Not enough evidence" in w.value for w in at.warning)
        assert at.error == []
        # It must never be presented as a confident grounded answer.
        assert not any("Grounded in the indexed paper" in s.value for s in at.success)

    def test_abstention_shows_no_sources_section(self):
        at = _run(
            indexed=INDEXED,
            backend_ok=True,
            answer=_abstained_answer(),
            asked_question="What was the inference latency?",
        )

        markdown = " ".join(block.value for block in at.markdown)
        assert "**Sources**" not in markdown


# ------------------------------------------------------- question validation
class TestQuestionValidation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("", False), ("   ", False), ("hello", True), (" hello ", True)],
    )
    def test_validity_uses_the_stripped_value(self, value, expected):
        assert frontend_app._question_is_valid(value) is expected

    @pytest.mark.parametrize(
        ("value", "color"),
        [
            ("", frontend_app.QUESTION_INVALID_COLOR),
            ("   ", frontend_app.QUESTION_INVALID_COLOR),
            ("hello", frontend_app.QUESTION_VALID_COLOR),
            (" hello ", frontend_app.QUESTION_VALID_COLOR),
        ],
    )
    def test_outline_tracks_the_current_text(self, value, color):
        at = _run(indexed=INDEXED, backend_ok=True)

        at.text_input[0].set_value(value).run()

        markup = " ".join(block.value for block in at.markdown)
        other = (
            frontend_app.QUESTION_VALID_COLOR
            if color == frontend_app.QUESTION_INVALID_COLOR
            else frontend_app.QUESTION_INVALID_COLOR
        )
        assert color in markup
        assert other not in markup


# --------------------------------------------------- active paper after index
class TestActivePaperAfterIndexing:
    def test_sidebar_shows_the_paper_as_soon_as_indexing_succeeds(self, monkeypatch):
        result = api_client.IndexResult(
            source="1512.03385v1.pdf",
            num_pages=12,
            num_chunks=70,
            total_vectors=70,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        monkeypatch.setattr(
            api_client,
            "ingest_pdf",
            lambda filename, data, **kwargs: {
                "source": filename,
                "num_pages": 12,
                "num_chunks": 70,
            },
        )
        monkeypatch.setattr(
            api_client, "index_pdf", lambda filename, data, **kwargs: result
        )

        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.session_state["backend_ok"] = True
        at.run()
        at.file_uploader[0].upload("1512.03385v1.pdf", b"%PDF-1.4 fake").run()

        next(
            button for button in at.button if button.label == "Ingest and index this paper"
        ).click().run()

        assert not at.exception
        sidebar = " ".join(block.value for block in at.sidebar.markdown)
        assert "No paper indexed yet." not in sidebar
        assert "1512.03385v1.pdf" in sidebar
        assert at.session_state["indexed"]["total_vectors"] == 70
