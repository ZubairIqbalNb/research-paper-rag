"""Grounded prompt construction for research-paper question answering.

Evidence blocks are numbered ``1..n`` in *reranked* order, i.e. ``rerank_rank + 1``.
That single invariant is what makes citations trustworthy: the ``[n]`` marker the
model is told to use always maps back to a real retrieved chunk.
"""
from __future__ import annotations

from backend.core.models import RerankedChunk

# Sentinel the model is asked to emit when the evidence cannot answer the question.
INSUFFICIENT_EVIDENCE_MARKER = "INSUFFICIENT_EVIDENCE"

_PROMPT_TEMPLATE = """You answer questions about research papers using ONLY the numbered evidence below.

Rules:
1. Use only the evidence. Never add facts from outside it.
2. Cite every claim with the bracketed numbers of the evidence you used, e.g. [1] or [2][3].
3. If the evidence does not answer the question, reply with {marker} on the first line, then one
   short sentence describing what is missing. Do not guess.
4. Be concise and technical, and keep the paper's terminology.

EVIDENCE
{evidence}

QUESTION: {query}
ANSWER:"""


def build_grounded_prompt(query: str, evidence: list[RerankedChunk]) -> str:
    """Build the grounded prompt from the reranked evidence.

    Raises:
        ValueError: The query is blank or there is no evidence to ground on.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query must be a non-empty string")
    if not evidence:
        raise ValueError("Cannot build a grounded prompt without evidence")

    blocks = [
        f"[{item.rerank_rank + 1}] source: {item.chunk.source} | page: {item.chunk.page_number}\n"
        f"{item.chunk.text}"
        for item in evidence
    ]

    return _PROMPT_TEMPLATE.format(
        marker=INSUFFICIENT_EVIDENCE_MARKER,
        evidence="\n\n".join(blocks),
        query=query,
    )


def is_insufficient_evidence(answer: str) -> bool:
    """Whether the model signalled insufficient evidence on its first line.

    Only the first line counts, so an ordinary answer that happens to contain the
    phrase (e.g. "the evidence is insufficient to conclude…") is not misread.
    """
    text = (answer or "").strip()
    if not text:
        return False
    first_line = text.splitlines()[0].strip()
    return first_line.upper().startswith(INSUFFICIENT_EVIDENCE_MARKER)


def strip_insufficient_evidence_marker(answer: str) -> str:
    """Drop the sentinel line, keeping the model's explanation (may be empty)."""
    lines = (answer or "").strip().splitlines()
    if lines and lines[0].strip().upper().startswith(INSUFFICIENT_EVIDENCE_MARKER):
        return "\n".join(lines[1:]).strip().strip(":-").strip()
    return (answer or "").strip()
