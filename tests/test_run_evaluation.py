"""Focused tests for the Phase 4 evaluation harness.

No network, no real models, no Gemini key: pipeline tests inject a
deterministic embedder, a fake reranker and a scripted generator through the
same constructors the harness itself uses, so only the harness's wiring and
scoring logic is under test. The locked dataset is only ever read.
"""
import json
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import evaluation.run_evaluation as ev  # noqa: E402
from conftest import _add_page  # noqa: E402  (tests dir is on sys.path via pytest.ini)
from fakes import DeterministicEmbedder, FakeReranker  # noqa: E402


# --------------------------------------------------------------------- helpers
def _write_pdf(path: Path, page_texts: list[str]) -> Path:
    """Build a real multi-page text PDF from page texts (reportlab via conftest)."""
    writer = PdfWriter()
    for text in page_texts:
        _add_page(writer, text)
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


def _any_pdf(tmp_path: Path) -> Path:
    """A small existing PDF for tests that only need a real file to hash."""
    return _write_pdf(tmp_path / "paper.pdf", ["Placeholder page text."])


class _Scripted:
    """Scripted generator double used by wiring/trace tests."""

    model_name = "scripted"

    def generate(self, prompt: str) -> str:
        return "Grounded answer based on [1]."


@pytest.fixture
def tmp_questions_path(tmp_path, monkeypatch):
    """Point the harness at a writable copy of the dataset for main() runs.

    The real questions.json is never touched: tests patch the module-level
    QUESTIONS_PATH/RESULTS_PATH globals instead.
    """
    copy = tmp_path / "questions.json"
    copy.write_text(ev.QUESTIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(ev, "QUESTIONS_PATH", copy)
    return copy


@pytest.fixture
def tmp_results_path(tmp_path, monkeypatch):
    results = tmp_path / "results" / "evaluation.json"
    monkeypatch.setattr(ev, "RESULTS_PATH", results)
    return results


# ------------------------------------------------------------------ validation
class TestLoadQuestions:
    def test_accepts_the_locked_dataset(self):
        questions = ev.load_questions()

        assert len(questions) == 15
        assert sum(1 for q in questions if q["answerable"]) == 12
        assert sum(1 for q in questions if not q["answerable"]) == 3

    def test_missing_file_fails_fast(self, tmp_path):
        with pytest.raises(ev.EvaluationInputError, match="not found"):
            ev.load_questions(tmp_path / "missing.json")

    def test_invalid_json_fails_fast(self, tmp_path):
        bad = tmp_path / "questions.json"
        bad.write_text("{not json", encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="valid JSON"):
            ev.load_questions(bad)

    def test_non_array_fails_fast(self, tmp_path):
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps({"id": "q01"}), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="JSON array"):
            ev.load_questions(bad)

    def test_wrong_total_fails_fast(self, tmp_path):
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(ev.load_questions()[:14]), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="exactly 15"):
            ev.load_questions(bad)

    def test_duplicate_ids_fail_fast(self, tmp_path):
        questions = ev.load_questions()
        questions[1]["id"] = questions[0]["id"]
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="Duplicate question id"):
            ev.load_questions(bad)

    def test_wrong_answerable_split_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        # Flip q01 to a well-formed unanswerable question: 11/4 instead of 12/3.
        questions[0]["answerable"] = False
        questions[0]["reference_answer"] = None
        questions[0]["expected_pages"] = []
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="12 answerable"):
            ev.load_questions(bad)

    def test_answerable_missing_reference_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        questions[0]["reference_answer"] = "   "
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="reference_answer"):
            ev.load_questions(bad)

    def test_answerable_missing_expected_pages_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        questions[0]["expected_pages"] = []
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="expected_pages"):
            ev.load_questions(bad)

    def test_unanswerable_with_reference_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        questions[12]["reference_answer"] = "not null"
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="reference_answer=null"):
            ev.load_questions(bad)

    def test_unanswerable_with_pages_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        questions[12]["expected_pages"] = [1]
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match=r"expected_pages=\[\]"):
            ev.load_questions(bad)

    def test_missing_required_key_fails_fast(self, tmp_path):
        questions = ev.load_questions()
        del questions[0]["category"]
        bad = tmp_path / "questions.json"
        bad.write_text(json.dumps(questions), encoding="utf-8")

        with pytest.raises(ev.EvaluationInputError, match="missing required keys"):
            ev.load_questions(bad)


# --------------------------------------------------------------------- scoring
class TestScoreRetrieval:
    def test_full_recall_and_hit_at_1(self):
        recall, hit1 = ev.score_retrieval([4, 5], [4, 5, 6])

        assert recall == 1.0
        assert hit1 is True

    def test_partial_recall_without_hit_at_1(self):
        recall, hit1 = ev.score_retrieval([4, 5], [6, 4])

        assert recall == 0.5
        assert hit1 is False

    def test_zero_recall(self):
        recall, hit1 = ev.score_retrieval([8], [1, 2, 3])

        assert recall == 0.0
        assert hit1 is False

    def test_hit_at_1_uses_rank_not_mere_presence(self):
        # Page 8 is present but not first: full recall, no hit@1 credit.
        recall, hit1 = ev.score_retrieval([8], [4, 8])

        assert recall == 1.0
        assert hit1 is False

    def test_empty_expected_pages_scores_zero(self):
        recall, hit1 = ev.score_retrieval([], [1, 2])

        assert recall == 0.0
        assert hit1 is False


class TestAbstention:
    def test_unanswerable_abstaining_is_correct(self):
        assert ev.score_abstention(False, grounded=False, generated_answer="not reported") is True

    def test_unanswerable_insufficient_marker_counts_as_abstention(self):
        assert (
            ev.score_abstention(
                False, grounded=True, generated_answer="INSUFFICIENT_EVIDENCE\nmissing data"
            )
            is True
        )

    def test_unanswerable_hallucinating_is_wrong(self):
        assert ev.score_abstention(False, grounded=True, generated_answer="ResNet-152 is 3.2 ms") is False

    def test_answerable_grounded_answer_is_correct(self):
        assert ev.score_abstention(True, grounded=True, generated_answer="A real answer [1].") is True

    def test_answerable_abstaining_is_wrong(self):
        assert ev.score_abstention(True, grounded=False, generated_answer="not reported") is False


class TestAnswerOverlap:
    def test_identical_text_scores_one(self):
        reference = "Layers learn a residual mapping with identity shortcuts."
        assert ev.score_answer(reference, reference) == 1.0

    def test_disjoint_text_scores_zero(self):
        assert ev.score_answer("residual mapping optimization", "photosynthesis chloroplasts") == 0.0

    def test_partial_overlap_is_partial(self):
        reference = "deeper plain networks suffer degradation and higher training error"
        generated = "deeper networks suffer degradation"
        score = ev.score_answer(reference, generated)

        assert 0.0 < score < 1.0

    def test_stop_words_do_not_inflate_the_score(self):
        assert ev.score_answer("the and of", "the and of") == 0.0


# --------------------------------------------------------------------- wiring
class TestBuildRealServices:
    def test_production_defaults_when_no_doubles_are_injected(self, tmp_path):
        """No DI arguments -> the same lazy production defaults as /ask."""
        from backend.rag.service import RagService
        from backend.retrieval.service import RetrievalService

        pdf = _write_pdf(tmp_path / "paper.pdf", ["ResNet paper text page one."])
        service = ev.build_real_services(pdf)

        assert isinstance(service, RagService)
        assert isinstance(service._retrieval, RetrievalService)
        assert service._retrieval.store_dir == ev.EVALUATION_STORE_DIR
        # Production defaults sit behind lazy properties; nothing is loaded.
        assert service._reranker is None and service._generator is None

    def test_injected_doubles_flow_through(self, tmp_path):
        pdf = _write_pdf(tmp_path / "paper.pdf", ["Some text."])
        embedder, reranker = DeterministicEmbedder(), FakeReranker()
        service = ev.build_real_services(
            pdf, generator=_Scripted(), embedder=embedder, reranker=reranker
        )

        assert service._retrieval.embedder is embedder
        assert service._reranker is reranker

        result = service.answer("some text about the paper")

        assert result.grounded is True
        assert result.llm_model == "scripted"
        assert reranker.calls  # the fake reranker was actually used


class TestPipelineTrace:
    def test_run_real_pipeline_captures_trace(self, tmp_path):
        pdf = _write_pdf(tmp_path / "paper.pdf", ["ResNet text one.", "ResNet text two."])
        service = ev.build_real_services(
            pdf,
            generator=_Scripted(),
            embedder=DeterministicEmbedder(),
            reranker=FakeReranker(),
        )

        trace, latency = ev.run_real_pipeline(service, "ResNet text one")

        assert trace["grounded"] is True
        assert trace["llm_model"] == "scripted"
        assert trace["generated_answer"] == "Grounded answer based on [1]."
        assert trace["retrieved"] and trace["reranked"] and trace["citations"]
        assert latency > 0.0
        assert {c["page_number"] for c in trace["retrieved"]} == {1, 2}
        # Rank order is preserved in the captured traces.
        assert len(trace["reranked"]) == len(trace["retrieved"])


# --------------------------------------------------------------------- report
class TestReport:
    def _result(self, *, answerable=True, recall=1.0, hit1=True, abstained=False, overlap=0.9):
        return ev.PerQuestionResult(
            id="q",
            category="c",
            question="q?",
            answerable=answerable,
            reference_answer="ref" if answerable else None,
            expected_pages=[1] if answerable else [],
            llm_model="m",
            generated_answer="a",
            grounded=not abstained,
            abstained=abstained,
            correct_abstention=(not abstained) if answerable else abstained,
            retrieved_pages=[1],
            reranked_pages=[1],
            retrieved_chunks=[],
            reranked_chunks=[],
            citations=[],
            retrieval_recall=recall if answerable else 0.0,
            retrieval_hit_at_1=hit1 if answerable else False,
            answer_overlap=overlap if answerable else None,
            answer_correct=(overlap >= ev.ANSWER_OVERLAP_THRESHOLD) if answerable else None,
            latency_seconds=1.0,
        )

    def test_summary_aggregates_answerable_metrics(self, tmp_path):
        results = [self._result(recall=1.0, hit1=True), self._result(recall=0.5, hit1=False)]
        report = ev.build_report(
            results,
            chunk_size=1000,
            chunk_overlap=200,
            source_pdf=_any_pdf(tmp_path),
            questions_sha256="0" * 64,
        )

        s = report.summary
        assert s.questions_run == 2
        assert s.answerable_questions == 2
        assert s.unanswerable_questions == 0
        assert s.retrieval_top1_hit == 1
        assert s.retrieval_any_hit == 2
        assert s.retrieval_macro_recall == 0.75
        assert s.answer_correct_count == 2
        assert s.abstained_on_answerable == 0
        assert s.avg_latency_seconds == 1.0
        assert s.pdf_sha256 == ev.file_sha256(_any_pdf(tmp_path))

    def test_summary_aggregates_abstention(self, tmp_path):
        results = [
            self._result(answerable=False, abstained=True),
            self._result(answerable=False, abstained=False),
        ]
        report = ev.build_report(
            results,
            chunk_size=1000,
            chunk_overlap=200,
            source_pdf=_any_pdf(tmp_path),
            questions_sha256="0" * 64,
        )

        s = report.summary
        assert s.unanswerable_questions == 2
        assert s.abstention_correct == 1
        assert s.abstention_total == 2
        assert s.answer_scored_count == 0

    def test_report_serialises_to_json(self, tmp_path, tmp_results_path):
        report = ev.build_report(
            [self._result()],
            chunk_size=1000,
            chunk_overlap=200,
            source_pdf=_any_pdf(tmp_path),
            questions_sha256="0" * 64,
        )
        ev.write_report(report, tmp_results_path)

        loaded = json.loads(tmp_results_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["questions_run"] == 1
        assert loaded["results"][0]["id"] == "q"


# --------------------------------------------------------------------- main()
class TestMain:
    def test_missing_questions_exits_2(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ev, "QUESTIONS_PATH", tmp_path / "missing.json")

        assert ev.main([]) == 2
        assert "not found" in capsys.readouterr().err

    def test_malformed_questions_exits_2(self, tmp_path, monkeypatch, capsys):
        bad = tmp_path / "questions.json"
        bad.write_text("{oops", encoding="utf-8")
        monkeypatch.setattr(ev, "QUESTIONS_PATH", bad)

        assert ev.main([]) == 2
        assert "valid JSON" in capsys.readouterr().err

    def test_missing_pdf_exits_2(self, tmp_questions_path, tmp_path, monkeypatch):
        monkeypatch.setattr(ev, "DEFAULT_PDF_CANDIDATES", [tmp_path / "nothing.pdf"])

        assert ev.main([]) == 2

    def test_explicit_pdf_not_found_exits_2(self, tmp_questions_path):
        assert ev.main(["--pdf", "/no/such/file.pdf"]) == 2

    def test_no_api_key_and_no_skip_flag_exits_2(
        self, tmp_questions_path, tmp_path, monkeypatch
    ):
        from backend.core.config import Settings

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(ev, "DEFAULT_PDF_CANDIDATES", [_any_pdf(tmp_path)])
        # get_settings() would pick up the gitignored .env; force keyless config.
        from backend.core.config import Settings

        monkeypatch.setattr(ev, "get_settings", lambda: Settings(gemini_api_key=None))

        assert ev.main([]) == 2

    def test_skip_api_end_to_end_writes_report(
        self, tmp_questions_path, tmp_results_path, tmp_path, monkeypatch
    ):
        pdf = _write_pdf(
            tmp_path / "paper.pdf",
            [
                "Residual learning eases degradation in deep plain networks on ImageNet.",
                "Shortcut connections add identity mappings to stacked layer outputs.",
            ],
        )
        monkeypatch.setattr(ev, "DEFAULT_PDF_CANDIDATES", [pdf])
        # Keep the end-to-end run model-free: real ingestion, retrieval and
        # scoring logic, but test doubles for the embedding/reranking models.
        # Capture the original first: the patch replaces the module attribute.
        original_build = ev.build_real_services
        monkeypatch.setattr(
            ev,
            "build_real_services",
            lambda *args, **kwargs: original_build(
                *args,
                embedder=DeterministicEmbedder(),
                reranker=FakeReranker(),
                **kwargs,
            ),
        )

        assert ev.main(["--skip-api"]) == 0

        report = json.loads(tmp_results_path.read_text(encoding="utf-8"))
        assert report["summary"]["questions_run"] == 15
        assert report["summary"]["answerable_questions"] == 12
        assert report["summary"]["unanswerable_questions"] == 3
        assert report["summary"]["llm_model"] == "offline-stub"
        assert report["summary"]["source_pdf"] == "paper.pdf"
        assert report["summary"]["questions_sha256"] == ev.file_sha256(tmp_questions_path)
        # Real ingestion + retrieval produced pages from the 2-page PDF.
        q01 = next(r for r in report["results"] if r["id"] == "q01")
        assert q01["retrieved_pages"] in ([1], [1, 2])
        # The offline stub always abstains, so answerable questions are flagged:
        # not grounded, no citations, abstention recorded.
        assert q01["grounded"] is False and q01["abstained"] is True
        assert q01["citations"] == []
        # --skip-api always abstains: the three unanswerables score correct...
        assert report["summary"]["abstention_correct"] == 3
        # ...and every answerable question is (correctly) flagged as abstaining.
        assert report["summary"]["abstained_on_answerable"] == 12

    def test_skip_api_never_touches_the_real_dataset(
        self, tmp_questions_path, tmp_results_path, tmp_path, monkeypatch
    ):
        before = tmp_questions_path.read_bytes()
        monkeypatch.setattr(ev, "DEFAULT_PDF_CANDIDATES", [_any_pdf(tmp_path)])

        assert ev.main(["--skip-api"]) == 0

        assert tmp_questions_path.read_bytes() == before
