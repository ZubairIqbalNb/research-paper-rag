"""RAG service tests: orchestration, evidence capture, citations, grounding."""
import pytest
from fakes import FakeGenerator, FakeReranker

from backend.llm.client import LLMError, GeminiClient
from backend.rag.prompt import INSUFFICIENT_EVIDENCE_MARKER
from backend.rag.service import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    NO_EVIDENCE_ANSWER,
    RagService,
)
from backend.reranking.reranker import CrossEncoderReranker
from backend.retrieval.vector_store import IndexNotBuiltError

CHUNK_SPECS = [
    ("retrieval evaluation metrics measure ranking quality of search systems", "paper_a.pdf", 1),
    ("self attention transformers process token sequences in parallel", "paper_a.pdf", 2),
    ("photosynthesis converts sunlight into chemical energy in plants", "paper_b.pdf", 7),
]
QUERY = "retrieval evaluation metrics ranking quality"


class _EmptyReranker:
    """Reranker double that drops all evidence (defensive no-evidence path)."""

    model_name = "empty-reranker"

    def rerank(self, query, chunks, top_n):  # noqa: ARG002 - protocol signature
        return []


def _indexed(make_chunks, retrieval_service):
    chunks = make_chunks(CHUNK_SPECS)
    retrieval_service.add_chunks(chunks)
    return chunks


# -------------------------------------------------------------- orchestration
def test_answer_runs_retrieval_rerank_and_generation(
    make_chunks, retrieval_service, rag_service, fake_reranker, fake_generator
):
    _indexed(make_chunks, retrieval_service)

    result = rag_service.answer(QUERY)

    # Retrieval: 20 candidates requested, all 3 indexed chunks returned.
    assert len(result.retrieved) == len(CHUNK_SPECS)
    # Reranking: the candidate list and top_n reached the reranker.
    query, candidates, top_n = fake_reranker.calls[0]
    assert query == QUERY
    assert len(candidates) == len(CHUNK_SPECS)
    assert top_n == 5
    assert len(result.reranked) == 3
    # Generation: exactly one prompt, grounded on the query and the evidence.
    assert len(fake_generator.prompts) == 1
    assert QUERY in fake_generator.prompts[0]
    assert result.answer == "Grounded answer based on [1]."
    assert result.grounded is True
    assert result.llm_model == fake_generator.model_name == "fake-gemini"


def test_answer_respects_candidates_and_top_n(
    make_chunks, retrieval_service, rag_service, fake_reranker
):
    _indexed(make_chunks, retrieval_service)

    result = rag_service.answer(QUERY, candidates=2, top_n=1)

    assert len(result.retrieved) == 2
    assert fake_reranker.calls[0][2] == 1
    assert len(result.reranked) == 1
    assert len(result.citations) == 1


def test_answer_preserves_both_orderings_and_both_scores(make_chunks, retrieval_service):
    """The Phase 2 cosine score survives reranking; ranks expose the movement."""
    chunks = _indexed(make_chunks, retrieval_service)
    # Force the worst-retrieved chunk to the top so a real swap is observable.
    reranker = FakeReranker(
        scores={
            chunks[2].text: 12.0,
            chunks[1].text: 4.0,
            chunks[0].text: 1.0,
        }
    )
    service = RagService(
        retrieval=retrieval_service, reranker=reranker, generator=FakeGenerator()
    )

    result = service.answer(QUERY)

    cosine_by_index = {chunk.chunk_index: chunk.score for chunk in result.retrieved}
    retrieval_order = [chunk.chunk_index for chunk in result.retrieved]
    promoted = result.reranked[0]
    assert promoted.chunk.text == chunks[2].text
    assert promoted.rerank_rank == 0
    # Reranking promoted a chunk that was NOT the best FAISS hit, and the
    # recorded retrieval_rank still points at its original position.
    assert promoted.retrieval_rank > 0
    assert retrieval_order[promoted.retrieval_rank] == promoted.chunk.chunk_index
    assert promoted.rerank_score == 12.0
    # Every reranked entry keeps its original cosine score untouched.
    for item in result.reranked:
        assert item.chunk.score == cosine_by_index[item.chunk.chunk_index]
    assert [item.rerank_rank for item in result.reranked] == [0, 1, 2]
    assert sorted(item.retrieval_rank for item in result.reranked) == [0, 1, 2]


def test_prompt_contains_only_the_reranked_evidence(
    make_chunks, retrieval_service, fake_generator
):
    chunks = _indexed(make_chunks, retrieval_service)
    reranker = FakeReranker(scores={chunks[2].text: 12.0})
    service = RagService(
        retrieval=retrieval_service, reranker=reranker, generator=fake_generator
    )

    service.answer(QUERY, top_n=1)

    prompt = fake_generator.prompts[0]
    assert chunks[2].text in prompt
    assert chunks[0].text not in prompt
    assert chunks[1].text not in prompt
    assert "paper_b.pdf" in prompt and "page: 7" in prompt


# ----------------------------------------------------------------- citations
def test_citations_come_from_chunk_metadata(make_chunks, retrieval_service, rag_service):
    chunks = _indexed(make_chunks, retrieval_service)

    result = rag_service.answer(QUERY)

    assert len(result.citations) == len(result.reranked)
    assert [citation.marker for citation in result.citations] == [
        item.rerank_rank + 1 for item in result.reranked
    ]
    by_index = {chunk.chunk_index: chunk for chunk in chunks}
    for citation, item in zip(result.citations, result.reranked):
        original = by_index[item.chunk.chunk_index]
        assert citation.source == original.source
        assert citation.page_number == original.page_number
        assert citation.chunk_index == original.chunk_index
        assert citation.rerank_score == item.rerank_score


def test_citations_are_empty_when_evidence_is_insufficient(
    make_chunks, retrieval_service
):
    chunks = _indexed(make_chunks, retrieval_service)
    generator = FakeGenerator(answer=f"{INSUFFICIENT_EVIDENCE_MARKER}\nNot reported.")
    service = RagService(
        retrieval=retrieval_service, reranker=FakeReranker(), generator=generator
    )

    result = service.answer(QUERY)

    assert result.citations == []
    assert result.grounded is False
    assert result.answer == "Not reported."
    assert chunks  # evidence was still retrieved and reranked
    assert result.reranked


# ------------------------------------------------------- insufficient evidence
def test_sentinel_without_explanation_uses_the_default_message(
    make_chunks, retrieval_service
):
    _indexed(make_chunks, retrieval_service)
    generator = FakeGenerator(answer=INSUFFICIENT_EVIDENCE_MARKER)
    service = RagService(
        retrieval=retrieval_service, reranker=FakeReranker(), generator=generator
    )

    result = service.answer(QUERY)

    assert result.grounded is False
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []


def test_answer_mentioning_insufficient_stays_grounded(
    make_chunks, retrieval_service
):
    _indexed(make_chunks, retrieval_service)
    generator = FakeGenerator(
        answer="The evidence available here is insufficient to rank methods [1]."
    )
    service = RagService(
        retrieval=retrieval_service, reranker=FakeReranker(), generator=generator
    )

    result = service.answer(QUERY)

    assert result.grounded is True
    assert result.citations
    assert generator.prompts  # the model was actually consulted


def test_no_evidence_short_circuits_without_calling_the_model(
    make_chunks, retrieval_service, fake_generator
):
    _indexed(make_chunks, retrieval_service)
    service = RagService(
        retrieval=retrieval_service, reranker=_EmptyReranker(), generator=fake_generator
    )

    result = service.answer(QUERY)

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.grounded is False
    assert result.reranked == [] and result.citations == [] and result.prompt == ""
    assert fake_generator.prompts == []  # never prompt a model with no evidence


# ------------------------------------------------------------ error propagation
def test_answer_propagates_index_not_built(retrieval_service, rag_service):
    with pytest.raises(IndexNotBuiltError):
        rag_service.answer(QUERY)


def test_answer_propagates_llm_errors(make_chunks, retrieval_service):
    _indexed(make_chunks, retrieval_service)
    generator = FakeGenerator(error=LLMError("Gemini request failed: boom"))
    service = RagService(
        retrieval=retrieval_service, reranker=FakeReranker(), generator=generator
    )

    with pytest.raises(LLMError, match="boom"):
        service.answer(QUERY)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidates": 0}, "candidates must be"),
        ({"candidates": 101}, "candidates must be <="),
        ({"top_n": 0}, "top_n must be >= 1"),
        ({"top_n": 21}, "top_n must be <="),
        ({"candidates": 2, "top_n": 3}, "top_n must be <= candidates"),
    ],
)
def test_answer_validates_limits(make_chunks, retrieval_service, rag_service, kwargs, message):
    _indexed(make_chunks, retrieval_service)

    with pytest.raises(ValueError, match=message):
        rag_service.answer(QUERY, **kwargs)


def test_answer_rejects_a_blank_query(rag_service):
    with pytest.raises(ValueError, match="non-empty"):
        rag_service.answer("   ")


# ------------------------------------------------------------------- wiring
def test_service_defaults_are_lazy_and_need_no_api_key(retrieval_service):
    service = RagService(retrieval=retrieval_service)

    # Defaults are constructed on first use and load no weights.
    assert isinstance(service.reranker, CrossEncoderReranker)
    assert service.reranker.is_loaded is False
    assert isinstance(service.generator, GeminiClient)
    assert service.generator.model_name  # configured, without touching the network
