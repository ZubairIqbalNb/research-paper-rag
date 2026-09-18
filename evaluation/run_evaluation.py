"""Phase 4 evaluation harness: score the real RAG pipeline against locked ground truth.

Runs every question in ``evaluation/questions.json`` through the REAL pipeline —
pdfplumber ingestion, MiniLM embeddings, FAISS retrieval, the ms-marco
cross-encoder reranker and Gemini generation — exactly as ``POST /ask`` would.
Nothing is bypassed or stubbed unless ``--skip-api`` is passed:

    .venv/bin/python evaluation/run_evaluation.py [--pdf PATH] [--skip-api]

Retrieval quality and answer quality are scored separately:

* retrieval: for answerable questions, did the expected pages make it into the
  reranked evidence that reached the generator (recall / hit@1)?
* abstention: for unanswerable questions, did the system take the pipeline's
  own not-grounded path (``INSUFFICIENT_EVIDENCE`` / no-evidence) instead of
  producing a hallucinated "grounded" answer?
* answer overlap: deterministic lexical overlap with the reference answer —
  an assistive heuristic for the report, never a substitute for reading it.

The only files written are ``evaluation/results/evaluation.json`` and, when
``--ragas`` is passed, ``evaluation/results/ragas.jsonl`` (plus the RAGAS block
merged into ``evaluation.json``). The ground-truth file is opened read-only,
validated eagerly and never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import get_settings  # noqa: E402
from backend.ingestion.chunker import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from backend.ingestion.pdf_parser import extract_pages  # noqa: E402
from backend.rag.prompt import is_insufficient_evidence  # noqa: E402
from backend.rag.service import (  # noqa: E402
    DEFAULT_RETRIEVAL_CANDIDATES,
    DEFAULT_TOP_N,
    RagService,
)
from backend.retrieval.service import RetrievalService  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "evaluation.json"

# The harness indexes the paper itself into a scratch store so runs are
# reproducible and never disturb the API's own index in data/faiss_store.
EVALUATION_STORE_DIR = PROJECT_ROOT / "data" / "faiss_store" / "evaluation_tmp"

DEFAULT_PDF_CANDIDATES = [
    Path("data/raw_pdfs/resnet.pdf"),
    Path("data/raw_pdfs/resnet_paper.pdf"),
    Path("data/raw_pdfs/paper.pdf"),
]

# Locked-dataset invariants (fail fast if the file ever deviates).
EXPECTED_TOTAL = 15
EXPECTED_ANSWERABLE = 12
EXPECTED_UNANSWERABLE = 3
REQUIRED_KEYS = {"id", "category", "question", "reference_answer", "answerable", "expected_pages"}

# Chunk text is kept in full (chunks are <= DEFAULT_CHUNK_SIZE chars) so the
# RAGAS layer can judge faithfulness / context precision on the real evidence.
_MAX_CHUNK_TEXT_CHARS = 1200
# A generated answer counts as "matching" when it recalls at least this share
# of the reference answer's content words.
ANSWER_OVERLAP_THRESHOLD = 0.5

# Model name reported by the deterministic ``--skip-api`` generator. The RAGAS
# layer refuses to score results carrying this name (see evaluation/ragas_eval.py).
OFFLINE_STUB_MODEL = "offline-stub"

_STOP_WORDS = frozenset(
    """
    the a an and or of to in on for with is are was were be been being that this
    these those it its as by at from into when their than not but if then so such
    can could may might will would should must do does did done have has had
    which what who whom whose how why where also more most other some any each
    per via using use used uses one two three both between within without over
    under about above below only same own very just
    """.split()
)


class EvaluationInputError(Exception):
    """The ground-truth dataset or a required input is missing or malformed."""


@dataclass
class PerQuestionResult:
    """One question's pipeline trace plus its scores."""

    id: str
    category: str
    question: str
    answerable: bool
    reference_answer: str | None
    expected_pages: list[int]
    llm_model: str
    generated_answer: str
    grounded: bool
    # Abstention = the pipeline's own not-grounded path was taken.
    abstained: bool
    # For unanswerable questions: abstained (did not hallucinate).
    # For answerable questions: did NOT abstain (gave a grounded answer).
    correct_abstention: bool
    # Distinct pages, for quick scanning; ranked traces live in the chunk lists.
    retrieved_pages: list[int]
    reranked_pages: list[int]
    # FAISS ordering (top candidates), rank order preserved.
    retrieved_chunks: list[dict]
    # Cross-encoder ordering (the evidence the generator actually saw).
    reranked_chunks: list[dict]
    citations: list[dict]
    # Share of expected pages present in the reranked evidence (answerable only).
    retrieval_recall: float
    # First reranked chunk sits on an expected page (answerable only).
    retrieval_hit_at_1: bool
    # Reference-token recall of the generated answer; None when unscored.
    answer_overlap: float | None
    answer_correct: bool | None
    latency_seconds: float


@dataclass
class EvaluationSummary:
    questions_run: int = 0
    answerable_questions: int = 0
    unanswerable_questions: int = 0
    retrieval_top1_hit: int = 0
    retrieval_any_hit: int = 0
    retrieval_macro_recall: float = 0.0
    abstention_correct: int = 0
    abstention_total: int = 0
    answer_correct_count: int = 0
    answer_scored_count: int = 0
    abstained_on_answerable: int = 0
    avg_latency_seconds: float = 0.0
    llm_model: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    retrieval_candidates: int = 0
    top_n: int = 0
    source_pdf: str = ""
    pdf_sha256: str = ""
    questions_sha256: str = ""


@dataclass
class EvaluationReport:
    summary: EvaluationSummary
    results: list[PerQuestionResult]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- dataset
def load_questions(path: Path | None = None) -> list[dict]:
    """Read and validate the LOCKED ground-truth dataset (strictly read-only).

    Raises:
        EvaluationInputError: The file is missing, unparseable, or deviates from
            the locked schema (count, split, unique ids, per-kind fields).
    """
    path = path or QUESTIONS_PATH
    if not path.is_file():
        raise EvaluationInputError(f"Ground-truth dataset not found: '{path}'")

    try:
        with open(path, encoding="utf-8") as fh:
            questions = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationInputError(f"Ground-truth dataset is not valid JSON: {exc}") from exc

    if not isinstance(questions, list):
        raise EvaluationInputError("Ground-truth dataset must be a JSON array of questions")
    if len(questions) != EXPECTED_TOTAL:
        raise EvaluationInputError(
            f"Expected exactly {EXPECTED_TOTAL} questions, found {len(questions)}"
        )

    seen_ids: set[str] = set()
    answerable_count = 0
    unanswerable_count = 0
    for index, question in enumerate(questions):
        label = f"questions[{index}]"
        if not isinstance(question, dict):
            raise EvaluationInputError(f"{label} must be a JSON object")
        missing = REQUIRED_KEYS - question.keys()
        if missing:
            raise EvaluationInputError(f"{label} is missing required keys: {sorted(missing)}")

        question_id = question["id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise EvaluationInputError(f"{label}.id must be a non-empty string")
        if question_id in seen_ids:
            raise EvaluationInputError(f"Duplicate question id: '{question_id}'")
        seen_ids.add(question_id)

        for key in ("category", "question"):
            if not isinstance(question[key], str) or not question[key].strip():
                raise EvaluationInputError(f"{label}.{key} must be a non-empty string")
        if not isinstance(question["answerable"], bool):
            raise EvaluationInputError(f"{label} ({question_id}): answerable must be a boolean")

        if question["answerable"]:
            answerable_count += 1
            reference = question["reference_answer"]
            pages = question["expected_pages"]
            if not isinstance(reference, str) or not reference.strip():
                raise EvaluationInputError(
                    f"{label} ({question_id}): answerable questions need a "
                    "non-empty reference_answer"
                )
            if not isinstance(pages, list) or not pages or not all(
                isinstance(page, int) and not isinstance(page, bool) and page >= 1
                for page in pages
            ):
                raise EvaluationInputError(
                    f"{label} ({question_id}): expected_pages must be a non-empty "
                    "list of 1-based page ints"
                )
        else:
            unanswerable_count += 1
            if question["reference_answer"] is not None:
                raise EvaluationInputError(
                    f"{label} ({question_id}): unanswerable questions must have "
                    "reference_answer=null"
                )
            if question["expected_pages"] != []:
                raise EvaluationInputError(
                    f"{label} ({question_id}): unanswerable questions must have "
                    "expected_pages=[]"
                )

    if (answerable_count, unanswerable_count) != (EXPECTED_ANSWERABLE, EXPECTED_UNANSWERABLE):
        raise EvaluationInputError(
            f"Locked dataset must contain {EXPECTED_ANSWERABLE} answerable and "
            f"{EXPECTED_UNANSWERABLE} unanswerable questions; found "
            f"{answerable_count} and {unanswerable_count}"
        )

    return questions


def file_sha256(path: Path) -> str:
    """Hex digest used to pin the exact inputs a run was produced from."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- pipeline
def find_source_pdf(explicit: Path | None = None) -> Path:
    """Locate the paper to index: an explicit ``--pdf`` wins, then defaults.

    Raises:
        EvaluationInputError: No candidate PDF exists.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise EvaluationInputError(f"--pdf not found: '{explicit}'")
        return explicit

    for candidate in DEFAULT_PDF_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in DEFAULT_PDF_CANDIDATES)
    raise EvaluationInputError(
        f"No source PDF found (searched: {searched}). Pass --pdf PATH."
    )


def build_real_services(
    pdf_path: Path,
    *,
    generator: object | None = None,
    embedder: object | None = None,
    reranker: object | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RagService:
    """Index the PDF and wire the real pipeline (no stubs, same defaults as /ask).

    Ingestion (pdfplumber + page-aware chunking), embeddings (the lazy default
    MiniLM embedder), FAISS persistence and the RAG service (lazy default
    cross-encoder reranker + Gemini generator) are all the production ones.
    ``generator``/``embedder``/``reranker`` are dependency-injection seams that
    default to the production components; only tests pass doubles.
    """
    pages = extract_pages(pdf_path, source_label=pdf_path.name)
    if not pages:
        raise EvaluationInputError(f"No extractable text found in '{pdf_path}'")

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    retrieval = RetrievalService(
        store_dir=EVALUATION_STORE_DIR,
        embedder=embedder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    retrieval.add_chunks(chunks, reset=True)

    return RagService(
        retrieval=retrieval,
        reranker=reranker,
        generator=generator,
        candidates=DEFAULT_RETRIEVAL_CANDIDATES,
        top_n=DEFAULT_TOP_N,
    )


def run_real_pipeline(service: RagService, question: str) -> tuple[dict, float]:
    """Run one question through the real RAG service, timed end to end."""
    started = time.perf_counter()
    result = service.answer(question)
    latency = time.perf_counter() - started

    retrieved = [
        {
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "score": round(float(chunk.score), 4),
            "text": chunk.text[:_MAX_CHUNK_TEXT_CHARS],
        }
        for chunk in result.retrieved
    ]
    reranked = [
        {
            "page_number": item.chunk.page_number,
            "chunk_index": item.chunk.chunk_index,
            "rerank_score": round(float(item.rerank_score), 4),
            "text": item.chunk.text[:_MAX_CHUNK_TEXT_CHARS],
        }
        for item in result.reranked
    ]
    citations = [asdict(citation) for citation in result.citations]

    return (
        {
            "llm_model": result.llm_model,
            "generated_answer": result.answer,
            "grounded": result.grounded,
            "retrieved": retrieved,
            "reranked": reranked,
            "citations": citations,
        },
        latency,
    )


# -------------------------------------------------------------------- scoring
def score_retrieval(
    expected_pages: list[int], ranked_pages: list[int]
) -> tuple[float, bool]:
    """Return (recall, hit@1) of expected pages within ranked evidence pages.

    ``ranked_pages`` must preserve evidence rank order (index 0 = best).
    """
    expected = {int(page) for page in expected_pages}
    if not expected:
        return 0.0, False
    ranked = [int(page) for page in ranked_pages]
    hits = expected.intersection(ranked)
    recall = len(hits) / len(expected)
    hit_at_1 = bool(ranked) and ranked[0] in expected
    return recall, hit_at_1


def is_abstention(grounded: bool, generated_answer: str) -> bool:
    """Whether the pipeline took one of its own not-grounded paths.

    ``RagResult.grounded`` is False exactly when the service returned the
    no-evidence answer or the model signalled ``INSUFFICIENT_EVIDENCE``; the
    marker check additionally catches a sentinel line on any path.
    """
    return (not grounded) or is_insufficient_evidence(generated_answer)


def score_abstention(answerable: bool, grounded: bool, generated_answer: str) -> bool:
    """Whether abstention behaviour was correct for this question kind.

    Unanswerable questions should abstain (not hallucinate a grounded answer);
    answerable questions should produce a grounded answer instead of abstaining.
    """
    abstained = is_abstention(grounded, generated_answer)
    return (not abstained) if answerable else abstained


def _content_tokens(text: str) -> list[str]:
    """Lowercase content words used for the deterministic overlap heuristic."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS and len(token) > 2
    ]


def score_answer(reference: str, generated: str) -> float:
    """Share of reference content words also present in the generated answer."""
    reference_tokens = set(_content_tokens(reference))
    if not reference_tokens:
        return 0.0
    generated_tokens = set(_content_tokens(generated))
    return len(reference_tokens & generated_tokens) / len(reference_tokens)


# ----------------------------------------------------------------- evaluation
def evaluate_question(service: RagService, question: dict) -> PerQuestionResult:
    """Run one ground-truth question through the pipeline and score the result."""
    pipeline, latency = run_real_pipeline(service, question["question"])

    retrieved_pages = [entry["page_number"] for entry in pipeline["retrieved"]]
    reranked_pages = [entry["page_number"] for entry in pipeline["reranked"]]

    answerable = bool(question["answerable"])
    grounded = pipeline["grounded"]
    generated = pipeline["generated_answer"]

    if answerable:
        recall, hit_at_1 = score_retrieval(question["expected_pages"], reranked_pages)
        overlap: float | None = round(score_answer(question["reference_answer"], generated), 4)
        answer_correct: bool | None = overlap >= ANSWER_OVERLAP_THRESHOLD
    else:
        recall, hit_at_1 = 0.0, False
        overlap, answer_correct = None, None

    return PerQuestionResult(
        id=question["id"],
        category=question["category"],
        question=question["question"],
        answerable=answerable,
        reference_answer=question["reference_answer"],
        expected_pages=question["expected_pages"],
        llm_model=pipeline["llm_model"],
        generated_answer=generated,
        grounded=grounded,
        abstained=is_abstention(grounded, generated),
        correct_abstention=score_abstention(answerable, grounded, generated),
        retrieved_pages=sorted(set(retrieved_pages)),
        reranked_pages=sorted(set(reranked_pages)),
        retrieved_chunks=pipeline["retrieved"],
        reranked_chunks=pipeline["reranked"],
        citations=pipeline["citations"],
        retrieval_recall=round(recall, 4),
        retrieval_hit_at_1=hit_at_1,
        answer_overlap=overlap,
        answer_correct=answer_correct,
        latency_seconds=round(latency, 3),
    )


def build_report(
    results: list[PerQuestionResult],
    *,
    chunk_size: int,
    chunk_overlap: int,
    source_pdf: Path,
    questions_sha256: str,
    notes: list[str] | None = None,
) -> EvaluationReport:
    """Aggregate per-question outcomes into the summary block."""
    answerable_results = [r for r in results if r.answerable]
    unanswerable_results = [r for r in results if not r.answerable]

    top1 = sum(1 for r in answerable_results if r.retrieval_hit_at_1)
    any_hit = sum(1 for r in answerable_results if r.retrieval_recall > 0)
    macro_recall = (
        sum(r.retrieval_recall for r in answerable_results) / len(answerable_results)
        if answerable_results
        else 0.0
    )

    summary = EvaluationSummary(
        questions_run=len(results),
        answerable_questions=len(answerable_results),
        unanswerable_questions=len(unanswerable_results),
        retrieval_top1_hit=top1,
        retrieval_any_hit=any_hit,
        retrieval_macro_recall=round(macro_recall, 4),
        abstention_correct=sum(1 for r in unanswerable_results if r.correct_abstention),
        abstention_total=len(unanswerable_results),
        answer_correct_count=sum(
            1 for r in answerable_results if r.answer_correct is True
        ),
        answer_scored_count=sum(
            1 for r in answerable_results if r.answer_correct is not None
        ),
        abstained_on_answerable=sum(1 for r in answerable_results if r.abstained),
        avg_latency_seconds=(
            round(sum(r.latency_seconds for r in results) / len(results), 3)
            if results
            else 0.0
        ),
        llm_model=results[0].llm_model if results else "",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_candidates=DEFAULT_RETRIEVAL_CANDIDATES,
        top_n=DEFAULT_TOP_N,
        source_pdf=source_pdf.name,
        pdf_sha256=file_sha256(source_pdf),
        questions_sha256=questions_sha256,
    )
    return EvaluationReport(summary=summary, results=results, notes=notes or [])


def write_report(report: EvaluationReport, path: Path | None = None) -> None:
    """Write the generated report (the only file the harness ever writes)."""
    path = path or RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(asdict(report), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 RAG evaluation harness")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to the source paper PDF")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Run without Gemini: retrieval/reranking stay real, generation is a "
        "deterministic offline stub (no API key needed).",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="After the pipeline run, also score the results with RAGAS "
        "(faithfulness / answer relevancy / context precision). Needs a real Gemini "
        "key and is incompatible with --skip-api.",
    )
    args = parser.parse_args(argv)

    if args.ragas and args.skip_api:
        print(
            "error: --ragas needs real generated answers; it cannot run with "
            "--skip-api.",
            file=sys.stderr,
        )
        return 2

    # Fail fast on any malformed/missing input before touching the pipeline.
    try:
        questions = load_questions(QUESTIONS_PATH)  # strictly read-only
        pdf_path = find_source_pdf(args.pdf)
    except EvaluationInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Source PDF: {pdf_path}")

    generator = None
    notes: list[str] = []
    if args.skip_api:
        class _OfflineGenerator:
            """Deterministic stand-in for Gemini; retrieval stays real."""

            model_name = OFFLINE_STUB_MODEL

            def generate(self, prompt: str) -> str:
                return "INSUFFICIENT_EVIDENCE\nOffline run: generation skipped (--skip-api)."

        generator = _OfflineGenerator()
        notes.append("Generation stubbed offline (--skip-api); retrieval/reranking are real.")
    elif not get_settings().has_gemini_api_key:
        print(
            "GEMINI_API_KEY is not configured; re-run with --skip-api for "
            "retrieval-only scoring.",
            file=sys.stderr,
        )
        return 2

    service = build_real_services(pdf_path, generator=generator)

    results: list[PerQuestionResult] = []
    for question in questions:
        print(f"[{question['id']}] {question['question'][:70]}...", flush=True)
        try:
            results.append(evaluate_question(service, question))
        except Exception as exc:
            print(f"  FAILED on {question['id']}: {exc}", file=sys.stderr)
            return 1

    report = build_report(
        results,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        source_pdf=pdf_path,
        questions_sha256=file_sha256(QUESTIONS_PATH),
        notes=notes,
    )
    write_report(report, RESULTS_PATH)
    print(f"Wrote {RESULTS_PATH}")

    if args.ragas:
        # Imported lazily: RAGAS stays an optional evaluation extra, and the
        # default pipeline path never pays its import cost.
        from evaluation import ragas_eval

        try:
            ragas_eval.run(RESULTS_PATH, ragas_eval.RAGAS_JSONL_PATH)
        except ragas_eval.RagasUnavailableError as exc:
            print(f"error: RAGAS layer unavailable: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {ragas_eval.RAGAS_JSONL_PATH}")

    summary = report.summary
    print(
        "\nSummary"
        f"\n  questions : {summary.questions_run} "
        f"({summary.answerable_questions} answerable / "
        f"{summary.unanswerable_questions} unanswerable)"
        f"\n  retrieval : top-1 hit {summary.retrieval_top1_hit}/"
        f"{summary.answerable_questions}, any hit {summary.retrieval_any_hit}/"
        f"{summary.answerable_questions}, macro recall "
        f"{summary.retrieval_macro_recall:.3f}"
        f"\n  abstention: {summary.abstention_correct}/{summary.abstention_total} correct"
        f"\n  answers   : {summary.answer_correct_count}/{summary.answer_scored_count} "
        f"match reference (lexical overlap >= {ANSWER_OVERLAP_THRESHOLD})"
        f"\n  latency   : {summary.avg_latency_seconds:.2f}s average"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
