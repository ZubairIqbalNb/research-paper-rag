"""Grounded prompt construction and insufficient-evidence sentinel tests."""
import pytest

from backend.core.models import RerankedChunk, RetrievedChunk
from backend.rag.prompt import (
    INSUFFICIENT_EVIDENCE_MARKER,
    build_grounded_prompt,
    is_insufficient_evidence,
    strip_insufficient_evidence_marker,
)

QUERY = "how is retrieval quality measured"

EVIDENCE_TEXT_A = "Retrieval quality is measured with nDCG and recall at k."
EVIDENCE_TEXT_B = "Attention weights are computed with a softmax over tokens."


def _item(text: str, *, rerank_rank: int, retrieval_rank: int, source: str, page: int) -> RerankedChunk:
    return RerankedChunk(
        chunk=RetrievedChunk(
            text=text,
            chunk_index=rerank_rank,
            page_number=page,
            source=source,
            score=0.7,
        ),
        rerank_score=9.0 - rerank_rank,
        retrieval_rank=retrieval_rank,
        rerank_rank=rerank_rank,
    )


@pytest.fixture
def evidence() -> list[RerankedChunk]:
    return [
        _item(EVIDENCE_TEXT_A, rerank_rank=0, retrieval_rank=2, source="paper_a.pdf", page=5),
        _item(EVIDENCE_TEXT_B, rerank_rank=1, retrieval_rank=0, source="paper_b.pdf", page=12),
    ]


def test_prompt_contains_the_question_and_the_grounding_rules(evidence):
    prompt = build_grounded_prompt(QUERY, evidence)

    assert f"QUESTION: {QUERY}" in prompt
    assert "using ONLY the numbered evidence" in prompt
    assert "Never add facts from outside it" in prompt
    assert "Cite every claim" in prompt


def test_prompt_numbers_evidence_in_reranked_order_with_source_and_page(evidence):
    prompt = build_grounded_prompt(QUERY, evidence)

    assert "[1] source: paper_a.pdf | page: 5" in prompt
    assert "[2] source: paper_b.pdf | page: 12" in prompt
    assert EVIDENCE_TEXT_A in prompt and EVIDENCE_TEXT_B in prompt
    # Marker [1] is the top reranked chunk even though it was retrieved second.
    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.index(EVIDENCE_TEXT_A) < prompt.index(EVIDENCE_TEXT_B)


def test_prompt_marker_follows_rerank_rank_not_retrieval_rank(evidence):
    prompt = build_grounded_prompt(QUERY, evidence)

    # Markers follow rerank_rank: the chunk retrieved first (retrieval_rank 0) is
    # marker 2 because it was reranked second.
    evidence_section = prompt.split("EVIDENCE", 1)[1]
    marker_one_block, marker_two_block = evidence_section.split("[2]", 1)
    assert EVIDENCE_TEXT_A in marker_one_block
    assert EVIDENCE_TEXT_B in marker_two_block


def test_prompt_instructs_the_insufficient_evidence_sentinel(evidence):
    prompt = build_grounded_prompt(QUERY, evidence)

    assert INSUFFICIENT_EVIDENCE_MARKER in prompt
    assert "first line" in prompt
    assert "Do not guess" in prompt


def test_prompt_is_deterministic(evidence):
    assert build_grounded_prompt(QUERY, evidence) == build_grounded_prompt(QUERY, evidence)


def test_prompt_rejects_blank_query_and_empty_evidence(evidence):
    with pytest.raises(ValueError, match="non-empty"):
        build_grounded_prompt("   ", evidence)
    with pytest.raises(ValueError, match="without evidence"):
        build_grounded_prompt(QUERY, [])


@pytest.mark.parametrize(
    "answer",
    [
        "INSUFFICIENT_EVIDENCE\nThe papers do not report this metric.",
        "insufficient_evidence",
        f"  {INSUFFICIENT_EVIDENCE_MARKER} on the first line is all I can say",
    ],
)
def test_is_insufficient_evidence_detects_the_first_line_marker(answer):
    assert is_insufficient_evidence(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "The retrieved evidence is insufficient to conclude that [1].",
        "nDCG is used to measure ranking quality [1].",
        f"nDCG is used to measure ranking quality [1].\n{INSUFFICIENT_EVIDENCE_MARKER}",
    ],
)
def test_is_insufficient_evidence_ignores_prose_and_later_lines(answer):
    assert is_insufficient_evidence(answer) is False


def test_strip_marker_keeps_the_explanation():
    answer = "INSUFFICIENT_EVIDENCE\nThe papers do not report this metric."

    assert strip_insufficient_evidence_marker(answer) == "The papers do not report this metric."


def test_strip_marker_returns_empty_when_there_is_nothing_else():
    assert strip_insufficient_evidence_marker("INSUFFICIENT_EVIDENCE") == ""
    assert strip_insufficient_evidence_marker("INSUFFICIENT_EVIDENCE:") == ""


def test_strip_marker_is_a_noop_for_normal_answers():
    answer = "nDCG measures ranking quality [1]."

    assert strip_insufficient_evidence_marker(answer) == answer
