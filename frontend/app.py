"""Research Paper Digest - Streamlit frontend (Phase 5).

A thin client over the existing FastAPI backend: upload a paper, index it, ask
grounded questions about it, then inspect the answer, its page citations and the
evidence that supported it. Every piece of RAG logic (ingestion, embeddings,
FAISS retrieval, cross-encoder reranking, the grounded Gemini prompt and
citations) lives behind the API - this file only renders UI and calls
``frontend.api_client``.

Run it with::

    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend import api_client  # noqa: E402

PAGE_TITLE = "Research Paper Digest"
PAGE_DESCRIPTION = (
    "Upload a research paper, index it, and ask questions that are answered "
    "**only** from that paper's own content - with page citations for every claim."
)
UNSUPPORTED_FILE_HINT = "Only PDF files are supported."

# Question-input validation. The outline must describe the text currently in the
# box - not whether a previous submission failed - so it is recomputed from the
# live widget value on every rerun and applied through the widget's
# ``st-key-<key>`` container class (see ``_question_outline_style``).
QUESTION_INPUT_KEY = "question_input"
QUESTION_VALID_COLOR = "#2e7d32"  # green outline for a usable question
QUESTION_INVALID_COLOR = "#c62828"  # red outline for empty/whitespace-only text


# --------------------------------------------------------------- session state
def _init_state() -> None:
    """Seed the small amount of UI state the app relies on."""
    st.session_state.setdefault("indexed", None)  # active indexed document
    st.session_state.setdefault("answer", None)  # latest /ask response
    st.session_state.setdefault("asked_question", None)


# ------------------------------------------------------------------- backend
def _check_backend() -> bool:
    """Return whether the FastAPI backend answers its health check."""
    try:
        return api_client.health()
    except Exception:  # any transport/HTTP problem just means "not reachable"
        return False


# --------------------------------------------------------------------- steps
def _handle_index(uploaded) -> None:
    """Ingest then index the uploaded PDF, reporting progress as it goes."""
    data = uploaded.getvalue()
    # A new paper invalidates any previous answer.
    st.session_state.answer = None
    st.session_state.asked_question = None

    with st.status("Processing paper...", expanded=True) as status:
        st.write(f"Extracting text from **{uploaded.name}**...")
        try:
            preview = api_client.ingest_pdf(uploaded.name, data)
        except api_client.BackendError as exc:
            status.update(label="Ingestion failed", state="error")
            st.error(str(exc))
            return
        except Exception:
            status.update(label="Ingestion failed", state="error")
            st.error("Unexpected error while ingesting this PDF.")
            return

        st.write(
            f"Read {preview.get('num_pages', 0)} pages into "
            f"{preview.get('num_chunks', 0)} chunks."
        )

        st.write("Embedding chunks and building the search index...")
        try:
            result = api_client.index_pdf(uploaded.name, data)
        except api_client.BackendError as exc:
            status.update(label="Indexing failed", state="error")
            st.error(str(exc))
            return
        except Exception:
            status.update(label="Indexing failed", state="error")
            st.error("Unexpected error while indexing this PDF.")
            return

        st.write(f"Indexed {result.total_vectors} vectors using `{result.model_name}`.")
        status.update(label="Paper ready", state="complete")

    st.session_state.indexed = {
        "source": result.source,
        "num_pages": result.num_pages,
        "num_chunks": result.num_chunks,
        "total_vectors": result.total_vectors,
        "model_name": result.model_name,
    }
    st.success(f"**{result.source}** is indexed and ready for questions.")


def _handle_ask(question: str) -> None:
    """Send one question to the backend and store the grounded response."""
    question = (question or "").strip()
    if not question:
        st.error("Please type a question first.")
        return

    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        try:
            answer = api_client.ask(question)
        except api_client.BackendError as exc:
            st.session_state.answer = None
            st.session_state.asked_question = None
            st.error(str(exc))
            return
        except Exception:
            st.session_state.answer = None
            st.session_state.asked_question = None
            st.error("Unexpected error while contacting the backend.")
            return

    st.session_state.answer = answer
    st.session_state.asked_question = question


# ---------------------------------------------------------------- validation
def _question_is_valid(question: str | None) -> bool:
    """A question is usable only when it has non-whitespace content."""
    return bool((question or "").strip())


def _question_outline_style(is_valid: bool) -> str:
    """Return CSS that outlines the question box green (valid) or red (invalid).

    Streamlit offers no native, server-rendered validation styling, and the
    colour has to follow the *current* text, so it is derived from the live
    widget value on each rerun and scoped to the question widget's
    ``st-key-<key>`` container.
    """
    border = QUESTION_VALID_COLOR if is_valid else QUESTION_INVALID_COLOR
    selector = f".st-key-{QUESTION_INPUT_KEY} div[data-baseweb='input']"
    return (
        "<style>"
        f"{selector} {{ border: 2px solid {border} !important;"
        " border-radius: 0.5rem; }"
        f"{selector}:focus-within {{ border-color: {border} !important;"
        " box-shadow: none; }"
        "</style>"
    )


# ------------------------------------------------------------------ rendering
def _render_sidebar() -> None:
    """Backend connection info and the active-document summary."""
    with st.sidebar:
        st.subheader("Backend")
        st.code(api_client.backend_url())
        if "backend_ok" not in st.session_state:
            st.session_state.backend_ok = _check_backend()
        if st.button("Check connection"):
            st.session_state.backend_ok = _check_backend()

        if st.session_state.backend_ok:
            st.success("Connected")
        else:
            st.error("Backend not reachable")

        st.caption(
            "Set `BACKEND_URL` before starting Streamlit to point at another "
            "FastAPI instance (default `http://localhost:8000`)."
        )

        st.divider()
        st.subheader("Active paper")
        indexed = st.session_state.get("indexed")
        if indexed:
            st.write(f"**{indexed['source']}**")
            st.caption(
                f"{indexed['num_pages']} pages · {indexed['num_chunks']} chunks · "
                f"{indexed['total_vectors']} vectors"
            )
            if st.button("Clear paper"):
                st.session_state.indexed = None
                st.session_state.answer = None
                st.session_state.asked_question = None
                st.rerun()
        else:
            st.write("No paper indexed yet.")


def _render_index_status() -> None:
    """Persistent status for the paper currently indexed."""
    indexed = st.session_state.get("indexed")
    if not indexed:
        return
    st.info(
        f"Indexed paper: **{indexed['source']}** - {indexed['num_pages']} pages, "
        f"{indexed['num_chunks']} chunks, {indexed['total_vectors']} vectors "
        f"(`{indexed['model_name']}`)."
    )


def _render_citations(answer: dict) -> None:
    """Show the pages that support the answer, straight from the API metadata."""
    citations = answer.get("citations") or []
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(
            f"- Page **{citation['page_number']}** · {citation['source']} "
            f"(evidence [{citation['marker']}])"
        )


def _render_evidence(answer: dict) -> None:
    """Expandable retrieval trace: reranked evidence first, then FAISS candidates."""
    reranked = answer.get("reranked") or []
    retrieved = answer.get("retrieved") or []
    if not reranked and not retrieved:
        return

    with st.expander("Retrieved Evidence", expanded=False):
        st.caption(
            f"{len(retrieved)} candidates retrieved from FAISS; {len(reranked)} "
            "kept after cross-encoder reranking. Bracketed markers match the "
            "citations above."
        )
        evidence_tab, candidates_tab = st.tabs(["Reranked evidence", "FAISS candidates"])

        with evidence_tab:
            if not reranked:
                st.write("No evidence survived reranking.")
            for item in reranked:
                st.markdown(
                    f"**[{item['rerank_rank'] + 1}] Page {item['page_number']}** · "
                    f"{item['source']} · chunk {item['chunk_index']}"
                )
                st.caption(
                    f"rerank score {item['rerank_score']:.3f} · "
                    f"cosine {item['score']:.3f}"
                )
                st.write(item["text"])
                st.divider()

        with candidates_tab:
            for rank, item in enumerate(retrieved, start=1):
                st.markdown(
                    f"**#{rank} Page {item['page_number']}** · {item['source']} · "
                    f"chunk {item['chunk_index']}"
                )
                st.caption(f"cosine score {item['score']:.3f}")
                st.write(item["text"])
                st.divider()


def _render_answer() -> None:
    """Render the latest answer, distinguishing grounded from abstained."""
    answer = st.session_state.get("answer")
    if not answer:
        return

    question = st.session_state.get("asked_question")
    st.divider()
    st.subheader("Answer")
    if question:
        st.caption(f"Question: {question}")

    if answer.get("grounded"):
        st.success("Grounded in the indexed paper.")
        st.markdown(answer.get("answer", ""))
    else:
        st.warning(
            "Not enough evidence: the indexed paper does not clearly support an "
            "answer to this question, so the system abstained instead of guessing."
        )
        st.markdown(answer.get("answer", ""))
        st.caption("This is an abstention, not an error.")

    _render_citations(answer)
    _render_evidence(answer)


# ----------------------------------------------------------------------- main
def main() -> None:
    """Compose the page: upload -> index -> ask -> answer."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📄", layout="centered")
    _init_state()

    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    st.header("1. Upload a paper")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help=UNSUPPORTED_FILE_HINT,
    )
    if uploaded is not None:
        size_kb = len(uploaded.getvalue()) / 1024
        st.caption(f"Selected: **{uploaded.name}** · {size_kb:.0f} KB")
        if st.button("Ingest and index this paper", type="primary"):
            _handle_index(uploaded)

    # Rendered after the upload/index step so a paper indexed on this very
    # interaction shows up in the sidebar immediately, without waiting for the
    # next rerun (or for a question to be asked).
    _render_sidebar()

    _render_index_status()

    st.header("2. Ask a question")
    if not st.session_state.get("indexed"):
        st.info("Upload and index a paper first, then ask questions about it.")
    else:
        # Deliberately outside ``st.form``: form widgets only commit on submit,
        # which would stop the outline below from tracking the text as the user
        # types or pastes. ``live`` commits on every change for instant feedback.
        question = st.text_input(
            "Your question about the paper",
            placeholder="e.g. What problem does this paper address?",
            key=QUESTION_INPUT_KEY,
            live="0ms",
        )
        st.markdown(
            _question_outline_style(_question_is_valid(question)),
            unsafe_allow_html=True,
        )
        if st.button("Ask", type="primary"):
            _handle_ask(question)

    _render_answer()

    st.divider()
    st.caption(
        "Answers come only from the indexed paper. When the evidence is "
        "insufficient, the backend abstains rather than guessing."
    )


if __name__ == "__main__":
    main()
