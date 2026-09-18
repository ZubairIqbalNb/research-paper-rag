"""RAG orchestration: retrieval -> reranking -> grounded prompt -> answer.

The only component that knows about all three collaborators. Each one is
injectable so the whole pipeline can be exercised without FAISS weights, cross
encoder weights or a Gemini API key.
"""
from __future__ import annotations

from backend.core.models import RagResult, RetrievedChunk
from backend.llm.client import AnswerGenerator, GeminiClient
from backend.reranking.reranker import Reranker, get_default_reranker
from backend.retrieval.service import RetrievalService
from backend.rag.citations import build_citations
from backend.rag.prompt import (
    build_grounded_prompt,
    is_insufficient_evidence,
    strip_insufficient_evidence_marker,
)

# Candidates retrieved from FAISS before reranking, and evidence kept afterwards.
DEFAULT_RETRIEVAL_CANDIDATES = 20
DEFAULT_TOP_N = 5
MAX_RETRIEVAL_CANDIDATES = 100
MAX_TOP_N = 20

NO_EVIDENCE_ANSWER = (
    "No indexed content matched this question, so there is nothing to answer from. "
    "Index a paper first (POST /index) or rephrase the question."
)
INSUFFICIENT_EVIDENCE_ANSWER = (
    "The indexed papers do not contain enough evidence to answer this question."
)


class RagService:
    """Answer questions grounded in the indexed papers."""

    def __init__(
        self,
        retrieval: RetrievalService,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
        candidates: int = DEFAULT_RETRIEVAL_CANDIDATES,
        top_n: int = DEFAULT_TOP_N,
    ) -> None:
        self._retrieval = retrieval
        self._reranker = reranker
        self._generator = generator
        self._candidates = candidates
        self._top_n = top_n

    # --------------------------------------------------------------- pipeline
    def answer(
        self,
        query: str,
        *,
        candidates: int | None = None,
        top_n: int | None = None,
    ) -> RagResult:
        """Retrieve, rerank, prompt and answer. Returns the full trace.

        Raises:
            ValueError: Invalid query or candidate/top_n values.
            IndexNotBuiltError: Nothing has been indexed yet.
            LLMError: The model call failed.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must be a non-empty string")

        candidates, top_n = self._resolve_limits(candidates, top_n)

        # 1. Wide semantic retrieval from FAISS (cosine scores, Phase 2 semantics).
        retrieved: list[RetrievedChunk] = self._retrieval.search(query, top_k=candidates)

        # 2. Narrow it down with the cross-encoder (logits).
        reranked = self.reranker.rerank(query, retrieved, top_n)

        # 3. Zero evidence: never prompt a model with nothing to ground on.
        if not reranked:
            return RagResult(
                query=query,
                retrieved=retrieved,
                reranked=[],
                answer=NO_EVIDENCE_ANSWER,
                grounded=False,
                citations=[],
                llm_model=self.generator.model_name,
                prompt="",
            )

        # 4. Grounded prompt over the reranked evidence, then the LLM answer.
        prompt = build_grounded_prompt(query, reranked)
        raw_answer = self.generator.generate(prompt)

        # 5. Insufficient evidence is a normal outcome, not an error.
        if is_insufficient_evidence(raw_answer):
            return RagResult(
                query=query,
                retrieved=retrieved,
                reranked=reranked,
                answer=strip_insufficient_evidence_marker(raw_answer) or INSUFFICIENT_EVIDENCE_ANSWER,
                grounded=False,
                citations=[],
                llm_model=self.generator.model_name,
                prompt=prompt,
            )

        return RagResult(
            query=query,
            retrieved=retrieved,
            reranked=reranked,
            answer=raw_answer.strip(),
            grounded=True,
            citations=build_citations(reranked),
            llm_model=self.generator.model_name,
            prompt=prompt,
        )

    # -------------------------------------------------------------- internals
    @property
    def reranker(self) -> Reranker:
        """The configured reranker, defaulting to the process-wide one."""
        if self._reranker is None:
            self._reranker = get_default_reranker()
        return self._reranker

    @property
    def generator(self) -> AnswerGenerator:
        """The configured answer generator, defaulting to the Gemini client.

        Creating the Gemini client is cheap and needs no API key; the key is only
        required when an answer is actually generated.
        """
        if self._generator is None:
            self._generator = GeminiClient()
        return self._generator

    def _resolve_limits(self, candidates: int | None, top_n: int | None) -> tuple[int, int]:
        candidates = self._candidates if candidates is None else candidates
        top_n = self._top_n if top_n is None else top_n

        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        if candidates > MAX_RETRIEVAL_CANDIDATES:
            raise ValueError(f"candidates must be <= {MAX_RETRIEVAL_CANDIDATES}")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if top_n > MAX_TOP_N:
            raise ValueError(f"top_n must be <= {MAX_TOP_N}")
        if top_n > candidates:
            raise ValueError("top_n must be <= candidates")
        return candidates, top_n
