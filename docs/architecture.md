# Architecture

Research Paper Digest is a retrieval-augmented question-answering system over a
single indexed research paper. It is split into a thin Streamlit frontend and a
FastAPI backend; all ingestion, retrieval, reranking, prompting and citation
logic lives in the backend.

This document describes what the repository actually implements. Where a
commonly expected RAG feature is **not** implemented, it is called out in
[Not implemented](#not-implemented).

## End-to-end flow

```
PDF upload (Streamlit)
        │  HTTP multipart
        ▼
POST /ingest  ──►  PDF validation ──► pdfplumber page extraction
        │               (page-aware chunking happens here and on /index)
        │
POST /index   ──►  same PDF → pages → chunks pipeline
        │               │
        │               ▼
        │        local embeddings (Sentence-Transformers MiniLM)
        │               │
        │               ▼
        │        FAISS inner-product index + JSON metadata sidecar (persisted)
        ▼
POST /ask     ──►  query embedding ──► FAISS retrieval (wide candidate set)
                        │
                        ▼
                 cross-encoder reranking (narrow top-N evidence)
                        │
                        ▼
                 grounded prompt (numbered evidence, reranked order)
                        │
                        ▼
                 Gemini generation
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        grounded answer      abstention (zero / insufficient evidence)
              │
              ▼
        metadata-derived citations ──► Streamlit answer + evidence UI
```

Stages that exist in code: PDF validation, page extraction, page-aware chunking,
local embeddings, FAISS indexing, query retrieval, cross-encoder reranking,
grounded prompting, Gemini generation, abstention, and metadata-based
citations. There is no query routing or decomposition between the query and
retrieval.

## System architecture

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| Frontend | `frontend/app.py`, `frontend/api_client.py` | Streamlit UI; HTTP client. No RAG logic. |
| API | `main.py`, `backend/api/*.py` | FastAPI app, routes, request validation, error mapping. |
| Ingestion | `backend/ingestion/*.py` | `pdfplumber` extraction and page-aware chunking. |
| Embeddings | `backend/embeddings/embedder.py` | Text → L2-normalized `float32` vectors. |
| Vector store | `backend/retrieval/vector_store.py` | FAISS index + JSON metadata, persistence. |
| Retrieval | `backend/retrieval/service.py` | Ties embedder + store together. |
| Reranking | `backend/reranking/reranker.py` | Cross-encoder scoring and top-N selection. |
| RAG | `backend/rag/*.py` | Orchestration, prompt building, citations. |
| LLM | `backend/llm/client.py` | Gemini text-in/text-out generation. |
| Config | `backend/core/config.py` | Environment-backed settings and secrets. |
| Evaluation | `evaluation/*.py` | Offline-capable harness plus optional RAGAS layer. |

### Frontend and backend responsibilities

The Streamlit app is a **thin client**. It never imports the retrieval, reranker
or LLM modules; it only calls the HTTP API through `frontend/api_client.py`. The
backend owns every model and every piece of state. The client talks to:

* `GET /health` — used for the sidebar connection indicator.
* `POST /ingest` — page/chunk preview before indexing.
* `POST /index` — embed and persist (called with `reset=true`, so the UI keeps a
  single active paper).
* `POST /ask` — grounded answer with citations and evidence.

The client also surfaces the backend's user-safe error messages (415/422/404/500/502/503)
and omits the internal prompt. The FastAPI app object is defined in `main.py`
as module-level `app` (so an ASGI server targets `main:app`); the client's
backend address defaults to `http://localhost:8000` and is overridable with the
`BACKEND_URL` environment variable.

## Ingestion

`backend/api/uploads.py` holds the shared "uploaded PDF → page-aware chunks"
pipeline used by both `/ingest` and `/index`:

1. **PDF validation.** The filename must end in `.pdf` or the multipart content
   type must be `application/pdf` (otherwise `415`). The bytes must begin with
   the `%PDF-` magic header (otherwise `422`). Chunk parameters are validated:
   `chunk_size > 0`, `chunk_overlap >= 0`, and `chunk_size >= chunk_overlap`
   (otherwise `422`).
2. **Page extraction.** `backend/ingestion/pdf_parser.py` uses `pdfplumber` to
   extract text page by page. Pages with no extractable text are skipped (so
   page numbers can be non-contiguous), `page_number` is **1-based** to match
   human citations, and `source` is the user-facing filename. A PDF with no
   extractable text at all (e.g. a scan) raises `422`.
3. **Page-aware chunking.** `backend/ingestion/chunker.py` uses LangChain's
   `RecursiveCharacterTextSplitter` (defaults `chunk_size=1000`,
   `chunk_overlap=200`, `length_function=len`) **per page**, so a chunk never
   spans a page boundary. `chunk_index` is a global ordinal across the document.

The result is a list of `Chunk` objects carrying `text`, `chunk_index`,
`page_number` and `source`. `POST /ingest` is stateless — it only previews
chunks and does not write anything to disk.

## Embeddings

`backend/embeddings/embedder.py` implements `SentenceTransformerEmbedder`:

* **Model:** `sentence-transformers/all-MiniLM-L6-v2` (default
  `DEFAULT_MODEL_NAME`), 384-dimensional, CPU by default, batch size 32.
* **Normalization:** vectors are requested normalized and then re-normalized
  (`_as_normalized_float32`) to guarantee unit-length rows, so inner product in
  FAISS equals cosine similarity. Zero vectors are left as zeros.
* **Lazy loading:** the model is imported and loaded only on first use
  (`load()`), so importing the module and booting the API stay cheap.
* `get_default_embedder()` caches one process-wide instance.
* The embedder is injectable (`Embedder` protocol), which is how tests run
  model-free.

There is no API-backed embedding provider; embeddings are always local.

## FAISS

`backend/retrieval/vector_store.py` implements `FaissVectorStore`:

* **Index type:** `faiss.IndexFlatIP` — an *exact* inner-product index. Because
  vectors are L2-normalized, inner product is cosine similarity. Scores are in
  `[-1, 1]`.
* **Metadata:** chunk metadata lives in a separate JSON sidecar
  (`metadata.json`), decoupled from the vectors. The invariant is positional:
  index row `i` ↔ `metadata["chunks"][i]`. `Chunk.chunk_index` is a per-document
  ordinal and is never used to address a vector.
* **Manifest:** `IndexManifest` stores `model_name`, `embedding_dim`,
  `num_vectors`, `normalized`, `chunk_size`, `chunk_overlap`, `schema_version`
  and `created_at`.
* **Persistence:** `save()` writes `index.faiss` and `metadata.json` to
  temporaries and then `os.replace`s both, so a failure never leaves a
  half-updated store. The default store directory is `data/faiss_store`
  (gitignored).
* **Load-time verification:** `load()` raises `IndexNotBuiltError` when no index
  exists and `IndexLoadError` for corruption/inconsistency — including a
  metadata/vector count mismatch, a dimension mismatch, an unknown-newer schema
  version, or an embedding-model mismatch against the configured model.
* **Append vs. replace:** `add()` appends and keeps row order; `build()` (used
  when `reset=true`) creates a fresh index. Mixing documents chunked with
  different parameters clears the stored chunk parameters to `None` to avoid
  ambiguity.

## Retrieval

`backend/retrieval/service.py` (`RetrievalService`) is the only component that
knows both the embedder and the store:

* `add_chunks(chunks, reset=...)` embeds chunk text, builds or appends to the
  store, saves it, and returns the manifest. Empty chunk lists are rejected.
* `search(query, top_k=5)` strips the query (blank → `ValueError`), embeds it,
  and returns `top_k` `RetrievedChunk` results ordered by descending cosine
  score. If no store exists it raises `IndexNotBuiltError` (`404`).
* FAISS is asked for `k = min(top_k, ntotal)`, so requesting more than the index
  holds returns fewer results rather than padded rows.

`POST /search` exposes this directly with `top_k` defaulting to `5`. `POST /ask`
uses it internally with a wider candidate count.

## Reranking

`backend/reranking/reranker.py` implements `CrossEncoderReranker`:

* **Model:** `cross-encoder/ms-marco-MiniLM-L6-v2` (default
  `DEFAULT_RERANK_MODEL`), CPU, batch size 16, lazily loaded.
* **Behavior:** every retrieved candidate is scored as a `(query, chunk)` pair
  jointly; results are sorted by descending score (a stable sort, so ties keep
  the original retrieval order) and truncated to `top_n`.
* **Score meaning:** `rerank_score` is the cross-encoder **logit** — higher means
  more relevant. It is not a probability, not normalized to `[0, 1]`, and is
  kept separate from the FAISS cosine `score`.
* **Ranks:** `retrieval_rank` is the item's original position in the FAISS
  ordering; `rerank_rank` is its position after reranking. Both are **0-based**
  internally and in the `/ask` JSON. Both are recorded on every `RerankedChunk`
  so the two orderings can be compared (see
  [`docs/evidence/rerank_before_after.json`](evidence/rerank_before_after.json)).
* **Defaults in the RAG path:** `DEFAULT_RETRIEVAL_CANDIDATES = 20` candidates
  retrieved, `DEFAULT_TOP_N = 5` kept as evidence (`MAX_RETRIEVAL_CANDIDATES = 100`,
  `MAX_TOP_N = 20`).
* `get_default_reranker()` caches one process-wide instance; the reranker is
  injectable so tests stay model-free.

## Grounded generation

`backend/rag/service.py` (`RagService.answer`) orchestrates:

1. Retrieval from FAISS (`candidates`, default 20).
2. Reranking to `top_n` (default 5).
3. Zero-evidence short-circuit (see below).
4. `build_grounded_prompt(query, reranked)` (`backend/rag/prompt.py`), then
   `generator.generate(prompt)`.
5. Insufficient-evidence handling, otherwise a grounded result with citations.

The prompt (`backend/rag/prompt.py`) numbers evidence blocks `1..n` in
**reranked order** (`rerank_rank + 1`) and instructs the model to use only that
evidence, cite every claim with bracketed numbers, be concise/technical, and
emit the sentinel `INSUFFICIENT_EVIDENCE` on the first line when the evidence
cannot answer. The prompt is kept on the result for testing/debugging and is
**not** exposed by `POST /ask`.

`backend/llm/client.py` (`GeminiClient`) is text-in/text-out: it reads the model
id and key from settings, uses `temperature=0.0`, creates the SDK client lazily,
and scrubs the API key out of error messages. It knows nothing about chunks or
citations, so the provider can be swapped without touching retrieval or
reranking. The default model is `gemini-3.8-flash`.

## Abstention

The service takes its own not-grounded path in two situations:

* **Zero evidence.** If reranking returns nothing, the model is *never* called;
  the service returns `NO_EVIDENCE_ANSWER`, `grounded=False`, and empty
  citations (`prompt=""`).
* **Insufficient evidence.** If the model's **first line** starts with
  `INSUFFICIENT_EVIDENCE` (`is_insufficient_evidence`), the result is
  `grounded=False`, citations are empty, and the answer is the remaining text
  (or `INSUFFICIENT_EVIDENCE_ANSWER` if that is empty). Only the first line is
  checked, so a normal answer that merely contains the phrase is not misread.

An abstention is a **successful 200 response**, not an error: the retrieved and
reranked evidence is still returned so the user can inspect what was considered.

## Citations

`backend/rag/citations.py` builds one `Citation` per evidence chunk directly from
`RerankedChunk`/`RetrievedChunk` metadata — `marker = rerank_rank + 1`, plus
`source`, `page_number`, `chunk_index` and `rerank_score`. The model's answer
text is **never parsed** for citations, so a citation can only reference a page
that was actually retrieved and shown to the model. The `[n]` markers the prompt
tells the model to use map back to precisely these chunks.

## Configuration and secrets

`backend/core/config.py` reads `GEMINI_API_KEY` and `GEMINI_MODEL` from the
environment (optionally via a gitignored `.env`; real environment variables take
precedence). The key is required only for `/ask`: when it is missing the endpoint
returns `503` *before* any model call. The key is never defaulted, never logged,
removed from the `Settings` `repr`, and scrubbed from error messages.
`LLMError` is mapped to `502`. See `.env.example` for the expected variables.

## Evaluation

`evaluation/run_evaluation.py` runs the **real** pipeline (pdfplumber ingestion,
MiniLM embeddings, FAISS retrieval, the ms-marco cross-encoder, and Gemini)
over the locked dataset in `evaluation/questions.json` (15 questions: 12
answerable, 3 unanswerable), indexing the paper into a scratch store
(`data/faiss_store/evaluation_tmp`) so runs are reproducible and never disturb
the API's own index.

* `--skip-api` swaps Gemini for a deterministic offline generator
  (`offline-stub`); **retrieval and reranking remain real**. This is the only way
  to run the harness without consuming Gemini quota.
* `--ragas` additionally runs the RAGAS layer (incompatible with `--skip-api`).
* The only files written are `evaluation/results/evaluation.json` and, with
  `--ragas`, `evaluation/results/ragas.jsonl`.

Scoring separates retrieval from answer quality:

* **Retrieval** (answerable): recall of `expected_pages` within the **reranked**
  evidence, and hit@1 (the first reranked chunk sits on an expected page).
* **Abstention** (unanswerable): did the pipeline take its own not-grounded path
  rather than producing a "grounded" answer.
* **Answer overlap**: deterministic content-word recall against the reference
  answer (`ANSWER_OVERLAP_THRESHOLD = 0.5`).

The committed `evaluation/results/evaluation.json` is an `--skip-api` run
(`llm_model: offline-stub`, note: *"Generation stubbed offline; retrieval/reranking
are real."*). Its real retrieval numbers are: 15 questions, 12 answerable /
3 unanswerable, top-1 hit 6/12, any hit 11/12, macro recall 0.875, abstention
3/3 correct. Because generation was stubbed, its `answer_correct_count` (0/12)
is **not** a meaningful measure of answer quality and must not be reported as
one.

`evaluation/ragas_eval.py` adds an optional RAGAS layer over already-produced
results. It uses the same Gemini model as the judge and adapts the project's
MiniLM embedder to RAGAS (no extra model). It records `faithfulness`,
`answer_relevancy`, and `context_precision` (the reference-based variant for
answerable questions, the reference-free variant for the 3 unanswerable ones),
and it **refuses to score** `--skip-api` results because those answers were never
really generated. No RAGAS results file is committed in this repository, so no
RAGAS scores are reported here.

## Not implemented

Verified absent from the codebase (do not attribute them to this project):

* Query routing, query decomposition, multi-query expansion, HyDE, or rewriting.
* Hybrid search (no BM25 / keyword retrieval); retrieval is pure dense FAISS.
* OCR for scanned/image-only PDFs (they are rejected with `422`).
* Multi-turn conversation memory or chat history.
* Streaming answers, authentication/authorization, or rate limiting.
* Any embedding/LLM provider other than the local MiniLM model and Gemini.
