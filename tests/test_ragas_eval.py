"""Focused tests for the RAGAS evaluation layer.

No network, no Gemini: the RAGAS metrics are replaced by small recording doubles
so only the layer's wiring, bookkeeping, aggregation and output are exercised.
These tests are NOT the real RAGAS evaluation - the live scores come from running
``evaluation/ragas_eval.py`` against the real judge model.
"""
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import evaluation.ragas_eval as ev  # noqa: E402
import evaluation.run_evaluation as run_ev  # noqa: E402


# --------------------------------------------------------------------- doubles
@dataclass
class _StubResult:
    """Stand-in for ragas' MetricResult (the layer only reads ``.value``)."""

    value: float


class _FakeMetric:
    """Recording double: captures kwargs and returns a fixed score or raises."""

    def __init__(self, value: float = 0.5, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[dict] = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _StubResult(self.value)


def _fake_metrics(
    *,
    faithfulness: float = 0.9,
    answer_relevancy: float = 0.8,
    context_precision: float = 0.7,
    error: Exception | None = None,
) -> ev.RagasMetrics:
    return ev.RagasMetrics(
        faithfulness=_FakeMetric(faithfulness, error),
        answer_relevancy=_FakeMetric(answer_relevancy, error),
        context_precision_with_reference=_FakeMetric(context_precision, error),
        context_precision_without_reference=_FakeMetric(context_precision, error),
        judge_model="fake-judge",
        embedding_model="fake-embed",
        library_version="0.0-test",
    )


def _result(**overrides) -> dict:
    base = {
        "id": "q01",
        "category": "methodology",
        "question": "Q?",
        "answerable": True,
        "reference_answer": "ref answer",
        "expected_pages": [1],
        "generated_answer": "generated [1]",
        "reranked_chunks": [
            {"page_number": 1, "chunk_index": 0, "rerank_score": 1.0, "text": "evidence one"}
        ],
        "retrieved_chunks": [
            {"page_number": 1, "chunk_index": 0, "score": 0.5, "text": "evidence one"}
        ],
        "citations": [],
    }
    base.update(overrides)
    return base


def _report(results: list[dict] | None = None) -> dict:
    return {
        "summary": {"questions_run": len(results or [_result()])},
        "results": results if results is not None else [_result()],
        "notes": [],
    }


# ------------------------------------------------------------------- contexts
class TestContextsFromResult:
    def test_prefers_reranked_evidence(self):
        result = _result(
            reranked_chunks=[{"text": "reranked"}],
            retrieved_chunks=[{"text": "retrieved"}],
        )

        assert ev.contexts_from_result(result) == ["reranked"]

    def test_falls_back_to_retrieved_when_not_reranked(self):
        result = _result(reranked_chunks=[], retrieved_chunks=[{"text": "retrieved"}])

        assert ev.contexts_from_result(result) == ["retrieved"]

    def test_skips_blank_text(self):
        result = _result(reranked_chunks=[{"text": "a"}, {"text": ""}, {"text": "b"}])

        assert ev.contexts_from_result(result) == ["a", "b"]

    def test_empty_when_no_chunks(self):
        assert ev.contexts_from_result(_result(reranked_chunks=[], retrieved_chunks=[])) == []


class TestPrecisionVariant:
    def test_answerable_uses_reference_variant(self):
        assert ev.precision_variant(_result()) == "with_reference"

    def test_unanswerable_uses_reference_free_variant(self):
        assert ev.precision_variant(_result(answerable=False, reference_answer=None)) == "without_reference"

    def test_blank_reference_uses_reference_free_variant(self):
        assert ev.precision_variant(_result(reference_answer="   ")) == "without_reference"


# -------------------------------------------------------------------- scoring
class TestScoreQuery:
    def test_wires_all_three_metrics(self):
        metrics = _fake_metrics(
            faithfulness=0.7, answer_relevancy=0.6, context_precision=0.5
        )

        out = ev.score_query(metrics, _result())

        assert out["scores"] == {
            "faithfulness": 0.7,
            "answer_relevancy": 0.6,
            "context_precision": 0.5,
        }
        assert out["context_precision_variant"] == "with_reference"
        assert out["errors"] == {}

        faithfulness_call = metrics.faithfulness.calls[0]
        assert faithfulness_call["user_input"] == "Q?"
        assert faithfulness_call["response"] == "generated [1]"
        assert faithfulness_call["retrieved_contexts"] == ["evidence one"]
        assert metrics.answer_relevancy.calls[0] == {"user_input": "Q?", "response": "generated [1]"}
        # The reference-based precision metric was used, not the reference-free one.
        assert metrics.context_precision_with_reference.calls
        assert metrics.context_precision_without_reference.calls == []

    def test_unanswerable_uses_reference_free_precision(self):
        metrics = _fake_metrics(context_precision=0.33)

        out = ev.score_query(
            metrics, _result(answerable=False, reference_answer=None)
        )

        assert out["context_precision_variant"] == "without_reference"
        assert out["scores"]["context_precision"] == 0.33
        assert metrics.context_precision_without_reference.calls
        assert metrics.context_precision_with_reference.calls == []

    def test_empty_answer_records_errors_instead_of_fake_scores(self):
        metrics = _fake_metrics()

        out = ev.score_query(metrics, _result(generated_answer=""))

        assert out["scores"]["faithfulness"] is None
        assert out["scores"]["answer_relevancy"] is None
        assert set(out["errors"]) == {"faithfulness", "answer_relevancy"}
        # Context precision needs no generated answer, so it is still scored.
        assert out["scores"]["context_precision"] == 0.7

    def test_metric_failure_is_recorded_per_metric(self):
        metrics = _fake_metrics(error=RuntimeError("judge down"))

        out = ev.score_query(metrics, _result())

        assert out["scores"] == {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }
        assert all("judge down" in reason for reason in out["errors"].values())
        assert len(out["errors"]) == 3

    def test_no_contexts_and_no_answer_errors_on_every_metric(self):
        metrics = _fake_metrics()

        out = ev.score_query(
            metrics,
            _result(generated_answer="", reranked_chunks=[], retrieved_chunks=[]),
        )

        assert out["scores"] == {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }
        assert set(out["errors"]) == {"faithfulness", "answer_relevancy", "context_precision"}


class TestAggregate:
    def test_means_ignore_unscored_rows_and_count_them(self):
        rows = [
            {"faithfulness": 1.0, "answer_relevancy": 0.0, "context_precision": None},
            {"faithfulness": 0.0, "answer_relevancy": 1.0, "context_precision": 0.5},
        ]

        agg = ev.aggregate(rows)

        assert agg["faithfulness"] == 0.5
        assert agg["answer_relevancy"] == 0.5
        assert agg["context_precision"] == 0.5
        assert agg["faithfulness_scored"] == 2
        assert agg["context_precision_scored"] == 1
        assert agg["num_queries"] == 2

    def test_all_unscored_gives_none_not_a_number(self):
        agg = ev.aggregate(
            [{"faithfulness": None, "answer_relevancy": None, "context_precision": None}]
        )

        assert agg["faithfulness"] is None
        assert agg["faithfulness_scored"] == 0


# --------------------------------------------------------------------- output
class TestWriteJsonl:
    def test_writes_one_object_per_line_and_creates_parent(self, tmp_path):
        jsonl = tmp_path / "nested" / "ragas.jsonl"
        rows = [{"id": "q01", "faithfulness": 0.5}, {"id": "q02", "faithfulness": 0.25}]

        ev.write_jsonl(rows, jsonl)

        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "q01"
        assert json.loads(lines[1])["faithfulness"] == 0.25


class TestEvaluateReport:
    def test_empty_results_raises(self):
        with pytest.raises(ev.RagasUnavailableError, match="no per-question results"):
            ev.evaluate_report({"results": []}, metrics_factory=_fake_metrics)

    def test_all_metrics_failing_raises_rather_than_reporting_nulls(self):
        metrics = _fake_metrics(error=RuntimeError("judge down"))

        with pytest.raises(ev.RagasUnavailableError, match="no scores for any query"):
            ev.evaluate_report(_report(), metrics_factory=lambda: metrics)

    def test_stubbed_offline_answers_are_refused(self):
        report = _report()
        report["summary"]["llm_model"] = run_ev.OFFLINE_STUB_MODEL

        with pytest.raises(ev.RagasUnavailableError, match="skip-api"):
            ev.evaluate_report(report, metrics_factory=lambda: _fake_metrics())

    def test_happy_path_returns_one_row_per_query(self):
        rows, metrics = ev.evaluate_report(
            _report([_result(), _result(id="q02")]),
            metrics_factory=lambda: _fake_metrics(faithfulness=0.4),
        )

        assert [row["id"] for row in rows] == ["q01", "q02"]
        assert rows[0]["faithfulness"] == 0.4
        assert rows[0]["retrieved_contexts"] == ["evidence one"]
        assert metrics.judge_model == "fake-judge"


# ---------------------------------------------------------------- end to end
class TestRun:
    def test_run_writes_jsonl_and_merges_block_without_touching_questions(
        self, tmp_path
    ):
        results = tmp_path / "evaluation.json"
        results.write_text(json.dumps(_report()), encoding="utf-8")
        jsonl = tmp_path / "ragas.jsonl"
        questions_before = run_ev.QUESTIONS_PATH.read_bytes()

        report = ev.run(results, jsonl, metrics_factory=lambda: _fake_metrics())

        rows = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert row["id"] == "q01"
        assert row["question"] == "Q?"
        assert row["generated_answer"] == "generated [1]"
        assert row["reference_answer"] == "ref answer"
        assert row["faithfulness"] == 0.9

        merged = json.loads(results.read_text(encoding="utf-8"))
        assert merged["ragas"]["library"] == "ragas"
        assert merged["ragas"]["metrics"] == list(ev.METRIC_NAMES)
        assert merged["ragas"]["aggregate"]["faithfulness"] == 0.9
        assert merged["results"][0]["ragas"]["answer_relevancy"] == 0.8

        # The locked ground-truth dataset is never touched.
        assert run_ev.QUESTIONS_PATH.read_bytes() == questions_before

    def test_run_writes_no_jsonl_when_everything_fails(self, tmp_path):
        results = tmp_path / "evaluation.json"
        results.write_text(json.dumps(_report()), encoding="utf-8")
        jsonl = tmp_path / "ragas.jsonl"
        metrics = _fake_metrics(error=RuntimeError("judge down"))

        with pytest.raises(ev.RagasUnavailableError):
            ev.run(results, jsonl, metrics_factory=lambda: metrics)

        assert not jsonl.exists()

    def test_missing_results_file_raises(self, tmp_path):
        with pytest.raises(ev.RagasUnavailableError, match="Results file not found"):
            ev.run(tmp_path / "nope.json", tmp_path / "ragas.jsonl")


# --------------------------------------------------------------------- wiring
class TestBuildEmbeddingAdapter:
    def test_adapter_round_trips_through_the_backend_embedder(self):
        class _Backend:
            model_name = "fake-mini"

            def embed(self, texts):
                return [[1.0, 2.0, 3.0] for _ in texts]

        adapter = ev.build_embedding_adapter(_Backend())

        assert adapter.embed_text("anything") == [1.0, 2.0, 3.0]
        assert asyncio.run(adapter.aembed_text("anything")) == [1.0, 2.0, 3.0]


class TestAsyncJudgeBridge:
    def test_agenerate_reuses_the_sync_judge_in_a_worker_thread(self):
        class _SyncJudge:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, response_model):
                self.calls.append((prompt, response_model))
                return "structured-result"

        inner = _SyncJudge()
        judge = ev._async_judge(inner)

        # RAGAS metrics call agenerate; the bridge must run the sync judge.
        assert asyncio.run(judge.agenerate("p1", "Model")) == "structured-result"
        assert judge.generate("p2", "Model") == "structured-result"
        assert inner.calls == [("p1", "Model"), ("p2", "Model")]


class TestBuildMetrics:
    def test_missing_api_key_fails_clearly_without_live_calls(self):
        from backend.core.config import Settings

        with pytest.raises(ev.RagasUnavailableError, match="GEMINI_API_KEY"):
            ev.build_metrics(settings=Settings(gemini_api_key=None))


# ----------------------------------------------------------------------- main
class TestMain:
    def test_missing_results_exits_2(self, tmp_path, capsys):
        assert ev.main(["--results", str(tmp_path / "missing.json")]) == 2
        assert "not found" in capsys.readouterr().err
