"""Cross-encoder reranking tests.

Fast tests inject a stub model (no weights). The real cross-encoder is covered
by the ``slow``-marked test at the bottom.
"""
import numpy as np
import pytest

from backend.core.models import RetrievedChunk
from backend.reranking.reranker import (
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    get_default_reranker,
)


class _StubCrossEncoder:
    """Duck-typed CrossEncoder scoring pairs from a text -> score mapping."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls: list[tuple[list[tuple[str, str]], dict]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(([tuple(pair) for pair in pairs], kwargs))
        return np.array(
            [self.scores_by_text[text] for _query, text in pairs], dtype=np.float64
        )


class _TwoColumnCrossEncoder(_StubCrossEncoder):
    """Stub that wrongly returns two scores per pair (e.g. a 2-label model)."""

    def predict(self, pairs, **kwargs):
        self.calls.append(([tuple(pair) for pair in pairs], kwargs))
        return np.ones((len(pairs), 2), dtype=np.float64)


def _chunk(text: str, index: int, *, source="paper.pdf", page=1, score=0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, chunk_index=index, page_number=page, source=source, score=score
    )


ALPHA = "alpha retrieval evaluation"
BETA = "beta attention transformers"
GAMMA = "gamma photosynthesis plants"

CHUNKS = [
    _chunk(ALPHA, 0, source="paper_a.pdf", page=3, score=0.91),
    _chunk(BETA, 1, source="paper_a.pdf", page=4, score=0.83),
    _chunk(GAMMA, 2, source="paper_b.pdf", page=9, score=0.72),
]


def _reranker(scores: dict[str, float], **kwargs) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name="stub-cross-encoder", model=_StubCrossEncoder(scores), **kwargs
    )


def test_rerank_orders_by_cross_encoder_score():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    results = reranker.rerank("query text", CHUNKS, top_n=3)

    assert [item.chunk.text for item in results] == [BETA, GAMMA, ALPHA]
    assert [item.rerank_score for item in results] == [5.0, 2.0, -1.0]
    assert [item.rerank_rank for item in results] == [0, 1, 2]


def test_rerank_records_both_orderings():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    results = reranker.rerank("query text", CHUNKS, top_n=3)

    # BETA was retrieved 2nd and reranked 1st => the demo-able "movement".
    by_text = {item.chunk.text: item for item in results}
    assert by_text[BETA].retrieval_rank == 1 and by_text[BETA].rerank_rank == 0
    assert by_text[ALPHA].retrieval_rank == 0 and by_text[ALPHA].rerank_rank == 2
    assert by_text[ALPHA].retrieval_rank - by_text[ALPHA].rerank_rank == -2


def test_rerank_preserves_all_retrieved_chunk_metadata():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    results = reranker.rerank("query text", CHUNKS, top_n=3)

    originals = {id(chunk): chunk for chunk in CHUNKS}
    assert len(results) == len(CHUNKS)
    for item in results:
        original = originals[id(item.chunk)]
        assert item.chunk is original
        assert item.chunk.source == original.source
        assert item.chunk.page_number == original.page_number
        assert item.chunk.chunk_index == original.chunk_index
        assert item.chunk.text == original.text


def test_rerank_never_overwrites_the_cosine_score():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    results = reranker.rerank("query text", CHUNKS, top_n=3)

    cosine_scores = {item.chunk.text: item.chunk.score for item in results}
    assert cosine_scores == {ALPHA: 0.91, BETA: 0.83, GAMMA: 0.72}
    # Rerank scores are separate (raw logits, may be negative).
    assert min(item.rerank_score for item in results) < 0


def test_rerank_truncates_to_top_n():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    results = reranker.rerank("query text", CHUNKS, top_n=2)

    assert [item.chunk.text for item in results] == [BETA, GAMMA]
    assert [item.rerank_rank for item in results] == [0, 1]


def test_rerank_top_n_above_candidate_count_returns_all():
    reranker = _reranker({BETA: 5.0, ALPHA: -1.0, GAMMA: 2.0})

    assert len(reranker.rerank("query text", CHUNKS, top_n=50)) == len(CHUNKS)


def test_rerank_keeps_retrieval_order_for_ties():
    reranker = _reranker({BETA: 7.0, ALPHA: 7.0, GAMMA: 7.0})

    results = reranker.rerank("query text", CHUNKS, top_n=3)

    assert [item.chunk.text for item in results] == [ALPHA, BETA, GAMMA]
    assert [item.retrieval_rank for item in results] == [0, 1, 2]


def test_rerank_empty_candidates_returns_empty_without_calling_the_model():
    model = _StubCrossEncoder({})
    reranker = CrossEncoderReranker(model_name="stub", model=model)

    assert reranker.rerank("query text", [], top_n=5) == []
    assert model.calls == []


def test_rerank_rejects_non_positive_top_n():
    reranker = _reranker({ALPHA: 1.0})

    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("query text", CHUNKS, top_n=0)


def test_rerank_sends_query_chunk_pairs_in_batches():
    model = _StubCrossEncoder({ALPHA: 1.0, BETA: 2.0, GAMMA: 3.0})
    reranker = CrossEncoderReranker(model_name="stub", model=model, batch_size=8)

    reranker.rerank("how is retrieval measured", CHUNKS, top_n=3)

    pairs, kwargs = model.calls[0]
    assert pairs == [("how is retrieval measured", chunk.text) for chunk in CHUNKS]
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is False


def test_rerank_rejects_models_that_return_multiple_scores_per_pair():
    reranker = CrossEncoderReranker(model_name="stub", model=_TwoColumnCrossEncoder({}))

    with pytest.raises(ValueError, match="one score per pair"):
        reranker.rerank("query text", CHUNKS, top_n=3)


def test_configured_default_rerank_model_is_cpu_marco_minilm():
    assert DEFAULT_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_reranker_model_is_loaded_lazily():
    reranker = CrossEncoderReranker()

    assert reranker.is_loaded is False
    assert get_default_reranker() is get_default_reranker()
    assert get_default_reranker().is_loaded is False
    assert reranker.model_name == DEFAULT_RERANK_MODEL


@pytest.mark.slow
def test_real_cross_encoder_ranks_the_relevant_passage_first():
    pytest.importorskip("sentence_transformers")
    reranker = CrossEncoderReranker(model_name=DEFAULT_RERANK_MODEL, device="cpu")

    relevant = "Retrieval quality is measured with nDCG, recall at k and mean reciprocal rank."
    irrelevant = "Photosynthesis converts sunlight and carbon dioxide into glucose."
    chunks = [
        _chunk(irrelevant, 0, score=0.55),
        _chunk(relevant, 1, score=0.51),
    ]

    first_run = reranker.rerank("how is retrieval quality measured", chunks, top_n=2)
    second_run = reranker.rerank("how is retrieval quality measured", chunks, top_n=2)

    assert first_run[0].chunk.text == relevant
    assert first_run[0].rerank_score > first_run[1].rerank_score
    assert first_run[0].retrieval_rank == 1 and first_run[0].rerank_rank == 0
    # Deterministic across runs, and logits are floats (not a crashed/NaN path).
    assert [item.rerank_score for item in first_run] == [
        item.rerank_score for item in second_run
    ]
    assert all(np.isfinite(item.rerank_score) for item in first_run)
