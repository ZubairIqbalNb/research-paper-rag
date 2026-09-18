"""RAGAS evaluation layer for the Phase 4 harness.

Complements (never replaces) the retrieval / answer-quality scoring in
``run_evaluation.py``. RAGAS scores the *already produced* pipeline outputs, so a
full live pipeline run is not needed to obtain these metrics:

    .venv/bin/python evaluation/ragas_eval.py            # scores results/evaluation.json
    .venv/bin/python evaluation/run_evaluation.py --ragas  # pipeline + RAGAS in one go

Three RAGAS metrics are recorded per query:

* ``faithfulness``      - is the generated answer supported by the retrieved evidence?
* ``answer_relevancy``  - does the answer actually address the question?
* ``context_precision`` - is the retrieved evidence relevant / well ranked?

The judge LLM is the *same* Gemini model the pipeline uses, built from the existing
``GEMINI_API_KEY`` / ``GEMINI_MODEL`` settings (the key is read from the environment
only, never logged). ``answer_relevancy`` needs embeddings, so the project's own
MiniLM embedder is adapted to RAGAS's interface - no extra model is introduced.

Ground truth in ``evaluation/questions.json`` is immutable and never read or written
by this module; only the generated ``evaluation/results/*`` files are touched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Privacy/offline friendliness: never send RAGAS telemetry, even before importing it.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("RAGAS_ANALYTICS_ENABLED", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "evaluation.json"
RAGAS_JSONL_PATH = Path(__file__).resolve().parent / "results" / "ragas.jsonl"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


class RagasUnavailableError(Exception):
    """RAGAS, its judge LLM, or its input results are missing or unusable."""


@dataclass
class RagasMetrics:
    """The RAGAS metrics wired to a judge LLM (and embeddings for relevancy)."""

    faithfulness: Any
    answer_relevancy: Any
    context_precision_with_reference: Any
    context_precision_without_reference: Any
    judge_model: str
    embedding_model: str
    library_version: str


# ------------------------------------------------------------------- contexts
def contexts_from_result(result: dict) -> list[str]:
    """Return the evidence the generator actually saw, in rank order.

    Prefers the reranked evidence (what reached the prompt) and falls back to the
    raw retrieval hits when reranking produced nothing. Text is preserved as
    stored so RAGAS judges the same context the model was grounded on.
    """
    chunks = result.get("reranked_chunks") or result.get("retrieved_chunks") or []
    return [chunk["text"] for chunk in chunks if chunk.get("text")]


def precision_variant(result: dict) -> str:
    """Which context-precision flavour applies: reference-based or not.

    The 12 answerable questions have a ground-truth reference, so context
    precision is judged against it. The 3 unanswerable questions have no
    reference (``reference_answer`` is null), so the reference-free variant is
    used and clearly recorded.
    """
    return "with_reference" if (result.get("reference_answer") or "").strip() else "without_reference"


# ---------------------------------------------------------------------- judge
def build_judge_llm(settings: Any = None) -> Any:
    """Build the RAGAS judge LLM from the project's existing Gemini settings.

    Raises:
        RagasUnavailableError: No usable ``GEMINI_API_KEY`` is configured, or
            RAGAS cannot be imported.
    """
    try:
        from ragas.llms import llm_factory
    except Exception as exc:  # pragma: no cover - import guard
        raise RagasUnavailableError(f"RAGAS is not importable: {exc}") from exc

    if settings is None:
        from backend.core.config import get_settings

        settings = get_settings()
    if not settings.has_gemini_api_key:
        raise RagasUnavailableError(
            "GEMINI_API_KEY is not configured; RAGAS needs an LLM judge. Set it "
            "in the environment or in a gitignored .env file and re-run."
        )

    from google import genai

    # The key stays local to the client; it is never logged or written to disk.
    client = genai.Client(api_key=settings.gemini_api_key)
    llm = llm_factory(settings.gemini_model, provider="google", client=client)
    return _async_judge(llm)


def _async_judge(llm: Any) -> Any:
    """Adapt a sync RAGAS judge to the async path its metrics actually use.

    RAGAS' collections metrics drive ``agenerate``, but instructor's google
    adapter exposes only a synchronous client, so a sync judge raises
    "Cannot use agenerate() with a synchronous client". Running the sync call in
    a worker thread satisfies both without changing which model is used.
    """
    from ragas.llms.base import InstructorBaseRagasLLM

    class _SyncToAsyncJudge(InstructorBaseRagasLLM):
        def generate(self, prompt: str, response_model: Any) -> Any:
            return llm.generate(prompt, response_model)

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            return await asyncio.to_thread(llm.generate, prompt, response_model)

    return _SyncToAsyncJudge()


def build_embedding_adapter(embedder: Any = None) -> Any:
    """Adapt the project's embedder to RAGAS's ``BaseRagasEmbedding`` interface."""
    from ragas.embeddings.base import BaseRagasEmbedding

    if embedder is None:
        from backend.embeddings.embedder import get_default_embedder

        embedder = get_default_embedder()

    class _ProjectEmbeddingAdapter(BaseRagasEmbedding):
        """Thin adapter: RAGAS calls ``embed_text``; the backend embeds a list."""

        def __init__(self, backend: Any) -> None:
            super().__init__()
            self._backend = backend

        def embed_text(self, text: str) -> list[float]:
            return [float(value) for value in self._backend.embed([text])[0]]

        async def aembed_text(self, text: str) -> list[float]:
            return self.embed_text(text)

    return _ProjectEmbeddingAdapter(embedder)


def build_metrics(settings: Any = None, embedder: Any = None) -> RagasMetrics:
    """Wire the three RAGAS metrics to the Gemini judge and the MiniLM embedder."""
    try:
        import ragas
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextPrecisionWithoutReference,
            Faithfulness,
        )
    except Exception as exc:  # pragma: no cover - import guard
        raise RagasUnavailableError(f"RAGAS is not importable: {exc}") from exc

    if settings is None:
        from backend.core.config import get_settings

        settings = get_settings()
    llm = build_judge_llm(settings)
    if embedder is None:
        from backend.embeddings.embedder import get_default_embedder

        embedder = get_default_embedder()
    embeddings = build_embedding_adapter(embedder)

    judge_model = getattr(settings, "gemini_model", None) or "unknown"
    return RagasMetrics(
        faithfulness=Faithfulness(llm=llm),
        answer_relevancy=AnswerRelevancy(llm=llm, embeddings=embeddings),
        context_precision_with_reference=ContextPrecisionWithReference(llm=llm),
        context_precision_without_reference=ContextPrecisionWithoutReference(llm=llm),
        judge_model=judge_model,
        embedding_model=getattr(embedder, "model_name", "unknown"),
        library_version=getattr(ragas, "__version__", "unknown"),
    )


# -------------------------------------------------------------------- scoring
def _score_value(metric: Any, **kwargs: Any) -> float:
    """Call one RAGAS metric and return its float value."""
    return round(float(metric.score(**kwargs).value), 4)


def score_query(metrics: RagasMetrics, result: dict) -> dict:
    """Score a single query with the three RAGAS metrics.

    Every metric call is recorded; a metric that cannot run (e.g. no evidence)
    is stored as ``None`` with an explicit reason, so nothing is silently skipped.
    """
    question = result["question"]
    response = (result.get("generated_answer") or "").strip()
    reference = (result.get("reference_answer") or "").strip()
    contexts = contexts_from_result(result)
    variant = precision_variant(result)

    scores: dict[str, float | None] = {name: None for name in METRIC_NAMES}
    errors: dict[str, str] = {}

    def attempt(name: str, call: Callable[[], float]) -> None:
        try:
            scores[name] = call()
        except Exception as exc:  # surfaced per metric, never hidden
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]

    if contexts and response:
        attempt(
            "faithfulness",
            lambda: _score_value(
                metrics.faithfulness,
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            ),
        )
    else:
        errors["faithfulness"] = "no retrieved contexts or empty generated answer"

    if response:
        attempt(
            "answer_relevancy",
            lambda: _score_value(
                metrics.answer_relevancy, user_input=question, response=response
            ),
        )
    else:
        errors["answer_relevancy"] = "empty generated answer"

    if not contexts:
        errors["context_precision"] = "no retrieved contexts"
    elif variant == "with_reference":
        attempt(
            "context_precision",
            lambda: _score_value(
                metrics.context_precision_with_reference,
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
    else:
        attempt(
            "context_precision",
            lambda: _score_value(
                metrics.context_precision_without_reference,
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            ),
        )

    return {
        "scores": scores,
        "context_precision_variant": variant,
        "errors": errors,
    }


def aggregate(per_query: list[dict]) -> dict:
    """Mean of each metric over the queries it could actually be scored on.

    ``per_query`` holds flat JSONL-shaped rows (one key per metric).
    """
    summary: dict[str, Any] = {"num_queries": len(per_query)}
    for name in METRIC_NAMES:
        values = [row[name] for row in per_query if row[name] is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
        summary[f"{name}_scored"] = len(values)
    return summary


def build_rows(report: dict, metrics: RagasMetrics) -> list[dict]:
    """Score every per-question result, returning one row per query (JSONL-shaped)."""
    rows: list[dict] = []
    for result in report.get("results", []):
        scored = score_query(metrics, result)
        rows.append(
            {
                "id": result["id"],
                "question": result["question"],
                "answerable": result["answerable"],
                "generated_answer": result["generated_answer"],
                "reference_answer": result.get("reference_answer"),
                "retrieved_contexts": contexts_from_result(result),
                **scored["scores"],
                "context_precision_variant": scored["context_precision_variant"],
                "errors": scored["errors"],
            }
        )
    return rows


# --------------------------------------------------------------------- output
def write_jsonl(rows: list[dict], path: Path | None = None) -> None:
    """Write one JSON object per line (the per-query metrics log)."""
    path = path or RAGAS_JSONL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_into_report(report: dict, rows: list[dict], metrics: RagasMetrics, jsonl_path: Path) -> dict:
    """Attach the RAGAS block to the report in place and return it."""
    by_id = {row["id"]: row for row in rows}
    for result in report.get("results", []):
        row = by_id.get(result["id"])
        if row is None:
            continue
        result["ragas"] = {
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["context_precision"],
            "context_precision_variant": row["context_precision_variant"],
            "errors": row["errors"],
        }

    errors = {row["id"]: row["errors"] for row in rows if row["errors"]}
    report["ragas"] = {
        "library": "ragas",
        "version": metrics.library_version,
        "judge_model": metrics.judge_model,
        "embedding_model": metrics.embedding_model,
        "metrics": list(METRIC_NAMES),
        "aggregate": aggregate(rows),
        "per_query_path": str(jsonl_path.relative_to(PROJECT_ROOT))
        if jsonl_path.is_relative_to(PROJECT_ROOT)
        else str(jsonl_path),
        "errors": errors,
    }
    return report


def _reject_stubbed_report(report: dict) -> None:
    """Refuse to score answers that were never really generated.

    The harness's ``--skip-api`` mode swaps Gemini for a deterministic stub, so
    any faithfulness / relevancy score computed from those answers would be
    meaningless. Rejecting them keeps misleading metrics out of the output.
    """
    from evaluation.run_evaluation import OFFLINE_STUB_MODEL

    model = (report.get("summary") or {}).get("llm_model")
    if model == OFFLINE_STUB_MODEL:
        raise RagasUnavailableError(
            f"Results were produced with --skip-api (generation stubbed as "
            f"'{OFFLINE_STUB_MODEL}'); RAGAS needs real generated answers. Re-run "
            "evaluation/run_evaluation.py with a Gemini key first."
        )


def evaluate_report(
    report: dict,
    *,
    metrics_factory: Callable[[], RagasMetrics] | None = None,
) -> tuple[list[dict], RagasMetrics]:
    """Score a loaded report, failing loudly if RAGAS could not score anything."""
    if not report.get("results"):
        raise RagasUnavailableError(
            "Results file has no per-question results to score; run "
            "evaluation/run_evaluation.py first."
        )
    _reject_stubbed_report(report)

    metrics = (metrics_factory or build_metrics)()
    rows = build_rows(report, metrics)

    scored = sum(1 for row in rows if any(row[name] is not None for name in METRIC_NAMES))
    if scored == 0:
        reasons = {row["id"]: row["errors"] for row in rows if row["errors"]}
        raise RagasUnavailableError(
            "RAGAS produced no scores for any query (no fabricated metrics are "
            f"written). First errors: {json.dumps(reasons)[:400]}"
        )
    return rows, metrics


def run(
    results_path: Path | None = None,
    jsonl_path: Path | None = None,
    *,
    metrics_factory: Callable[[], RagasMetrics] | None = None,
) -> dict:
    """Read the generated results, score them with RAGAS, and persist everything."""
    results_path = Path(results_path) if results_path else RESULTS_PATH
    jsonl_path = Path(jsonl_path) if jsonl_path else RAGAS_JSONL_PATH

    if not results_path.is_file():
        raise RagasUnavailableError(f"Results file not found: '{results_path}'")
    with open(results_path, encoding="utf-8") as fh:
        report = json.load(fh)

    rows, metrics = evaluate_report(report, metrics_factory=metrics_factory)
    write_jsonl(rows, jsonl_path)
    merge_into_report(report, rows, metrics, jsonl_path)

    with open(results_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS layer over the Phase 4 evaluation results")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--jsonl", type=Path, default=RAGAS_JSONL_PATH)
    args = parser.parse_args(argv)

    try:
        report = run(args.results, args.jsonl)
    except RagasUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: RAGAS evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    block = report["ragas"]
    agg = block["aggregate"]
    print(f"Wrote {args.jsonl}")
    print(
        "RAGAS summary"
        f"\n  library     : ragas {block['version']} (judge: {block['judge_model']})"
        f"\n  queries     : {agg['num_queries']}"
        f"\n  faithfulness: {agg['faithfulness']} ({agg['faithfulness_scored']} scored)"
        f"\n  relevancy   : {agg['answer_relevancy']} ({agg['answer_relevancy_scored']} scored)"
        f"\n  ctx precis. : {agg['context_precision']} ({agg['context_precision_scored']} scored)"
    )
    if block["errors"]:
        print(f"  warnings    : {len(block['errors'])} query(s) with per-metric errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
