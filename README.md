# Research Paper Digest

A retrieval-augmented (RAG) question-answering system for research papers. You
upload a PDF, the backend extracts and indexes it, and you ask questions that are
answered **only** from that paper's own content — with page/source citations for
every claim. When the indexed paper does not support an answer, the system
**abstains instead of guessing**.

A Streamlit UI drives a FastAPI backend over HTTP. All ingestion, embedding,
retrieval, reranking, prompting and citation logic lives in the backend; the UI
is a thin client.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [RAG pipeline](#rag-pipeline)
4. [Technology stack](#technology-stack)
5. [Repository structure](#repository-structure)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the application](#running-the-application)
9. [API reference](#api-reference)
10. [Grounding and abstention](#grounding-and-abstention)
11. [Citations](#citations)
12. [Evaluation](#evaluation)
13. [RAGAS](#ragas)
14. [Testing](#testing)
15. [Validation example](#validation-example)
16. [Known limitations](#known-limitations)
17. [Security and secrets](#security-and-secrets)
18. [Further documentation](#further-documentation)

---

## What it does

* **Ingest** a text-based PDF: validate it, extract page text, and split it into
  page-aware chunks.
* **Index** it: embed each chunk locally and persist a FAISS vector index plus a
  JSON metadata sidecar.
* **Ask** questions: retrieve a wide candidate set from FAISS, narrow it with a
  cross-encoder reranker, build a grounded prompt over the top evidence, and
  generate an answer with Gemini.
* **Cite**: return page/source citations derived from the retrieved chunk
  metadata (never parsed out of the model's prose).
* **Abstain**: refuse to answer when the evidence is insufficient.

The design goal is trustworthy answers scoped to a single paper, not a general
chat assistant.

---

## Architecture

```
   User
    │
    ▼
┌─────────────────────────────┐        HTTP (httpx)         ┌──────────────────────────────┐
│  Streamlit frontend         │  ───────────────────────►   │  FastAPI backend             │
│  frontend/app.py            │   GET  /health              │  health.py  (app = FastAPI)  │
│  frontend/api_client.py     │   POST /ingest              │  backend/api/*               │
│                             │   POST /index               │                              │
│  • upload + index UI        │   POST /ask                 │  ┌────────────────────────┐  │
│  • renders answer,          │  ◄───────────────────────   │  │ RAG services           │  │
│    citations, evidence      │        JSON responses        │  │ ingestion → embeddings │  │
│                             │                              │  │ → FAISS → reranking    │  │
│  Thin client: no            │                              │  │ → Gemini → citations   │  │
│  embeddings / retrieval /   │                              │  └────────────────────────┘  │
│  reranking / prompting      │                              │                              │
└─────────────────────────────┘                              └──────────────────────────────┘
```

**Separation of responsibilities**

| Component | Responsibility |
| --- | --- |
| Streamlit frontend | UI only: upload, index, ask, and display answers/citations/evidence. Talks to the backend over HTTP via `frontend/api_client.py`. |
| FastAPI backend | Owns every model and all state: validation, extraction, chunking, embeddings, FAISS, reranking, prompting, generation, citations. |

The Streamlit app does **not** embed text, query FAISS, rerank, build prompts, or
call Gemini directly — those all happen behind the API. The frontend only ever
receives JSON responses, and the Gemini API key never reaches it.

The FastAPI application object is defined in `health.py` as module-level `app`.

---

## RAG pipeline

```
PDF upload
   │
   ▼
PDF validation ──► page-by-page text extraction (pdfplumber)
   │
   ▼
page-aware chunking (each page split independently; chunks never cross pages)
   │
   ▼
local embeddings (sentence-transformers/all-MiniLM-L6-v2, L2-normalized)
   │
   ▼
FAISS inner-product index + JSON metadata sidecar (persisted to disk)
   │
   ▼
query retrieval  ──►  ~20 candidate chunks (cosine similarity)
   │
   ▼
cross-encoder reranking  ──►  5 evidence chunks (cross-encoder scores)
   │
   ▼
grounded prompt (numbered evidence, reranked order)
   │
   ▼
Gemini generation (temperature 0.0)
   │
   ├── evidence supports an answer ──► grounded answer + metadata citations
   └── evidence insufficient       ──► abstention (no fabricated answer)
```

**Actual defaults (from the source code)**

| Setting | Default | Where |
| --- | --- | --- |
| Chunk size | `1000` characters | `backend/ingestion/chunker.py` |
| Chunk overlap | `200` characters | `backend/ingestion/chunker.py` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | `backend/embeddings/embedder.py` |
| Vector index | FAISS `IndexFlatIP` (exact inner product = cosine on normalized vectors) | `backend/retrieval/vector_store.py` |
| Retrieval candidates | `20` (max `100`) | `backend/rag/service.py` |
| Reranking top-N evidence | `5` (max `20`) | `backend/rag/service.py` |
| `/search` default `top_k` | `5` | `backend/retrieval/service.py` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L6-v2` | `backend/reranking/reranker.py` |
| Gemini model | `gemini-3.8-flash` (temperature `0.0`) | `backend/core/config.py`, `backend/llm/client.py` |

Both models load lazily on first use and are injected as protocols, so tests run
without downloading weights or calling an API.

---

## Technology stack

Versions are pinned in `requirements.txt`.

| Area | Library | Version |
| --- | --- | --- |
| Language | Python | 3.12 (developed on 3.12.3) |
| Backend API | FastAPI | `0.141.1` |
| ASGI server | Uvicorn | `0.53.0` |
| File uploads | python-multipart | `0.0.20` |
| Frontend | Streamlit | `1.64.0` |
| HTTP client | httpx | `0.28.1` |
| PDF extraction | pdfplumber | `0.11.10` |
| Chunking | langchain-text-splitters | `1.1.2` |
| Embeddings / reranking | sentence-transformers | `6.0.1` |
| Vector search | faiss-cpu | `1.15.1` |
| Numerics | numpy | `2.5.3` |
| LLM SDK | google-genai | `2.24.0` |
| Config | python-dotenv | `1.2.3` |
| Testing | pytest | `9.1.1` (plus `pypdf`, `reportlab`) |
| Evaluation | ragas | `0.4.3` (plus `langchain-community`, `instructor`, `jsonref`, `openai`) |

CPU-only PyTorch is installed separately (see [Installation](#installation)).

---

## Repository structure

```
research-paper-rag/
├── health.py                  # FastAPI app: mounts routers, GET / and /health
├── requirements.txt
├── pytest.ini                 # pytest config + `slow` / `live` markers
├── .env.example               # environment variable template (no secrets)
├── backend/
│   ├── core/                  # shared models + environment-backed config
│   ├── ingestion/             # pdfplumber extraction + page-aware chunking
│   ├── embeddings/            # Sentence-Transformer embedder (MiniLM)
│   ├── retrieval/             # RetrievalService + FAISS store with JSON sidecar
│   ├── reranking/             # cross-encoder reranker
│   ├── llm/                   # Gemini text-in/text-out client
│   ├── rag/                   # orchestration, grounded prompt, citations
│   └── api/                   # /ingest, /index, /search, /ask routes
├── frontend/                  # Streamlit app + HTTP client (thin client)
├── evaluation/
│   ├── questions.json         # locked ground-truth dataset (15 questions)
│   ├── run_evaluation.py      # real-pipeline evaluation harness
│   ├── ragas_eval.py          # optional RAGAS layer
│   └── results/               # evaluation.json (and ragas.jsonl when run)
├── tests/                     # pytest suite (unit + API + frontend, live opt-in)
├── data/                      # gitignored runtime data
│   ├── raw_pdfs/              # source PDFs
│   └── faiss_store/           # persisted index.faiss + metadata.json
└── docs/                      # deeper architecture / API / evidence docs
```

---

## Installation

Requires Python 3.12 and an internet connection for the first model download and
PyTorch install.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the pinned dependencies
pip install -r requirements.txt

# 3. Install CPU-only PyTorch (kept separate so the default CUDA wheels and
#    multi-GB nvidia-* dependencies are not pulled on Linux)
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Create your environment file
cp .env.example .env               # then edit .env and add your GEMINI_API_KEY
```

On first use the two sentence-transformers models are downloaded into
`~/.cache/huggingface` (they are not committed to the repository).

---

## Configuration

Environment variables (see `.env.example`). Secrets belong in `.env`, which is
gitignored.

| Variable | Used by | Required | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Backend | Yes, for `/ask` | Google AI Studio API key. Read from the environment/`.env`, never logged. Without it, `POST /ask` returns `503` before any model call. |
| `GEMINI_MODEL` | Backend | No | Model id; defaults to `gemini-3.8-flash` when unset or blank. |
| `BACKEND_URL` | Frontend | No | Backend base URL for Streamlit; defaults to `http://localhost:8000`. |

Real environment variables take precedence over `.env` values.

---

## Running the application

### 1. Backend (FastAPI)

From the project root, with the virtual environment active:

```bash
uvicorn health:app --reload --port 8000
```

Verify it is up:

```bash
curl http://localhost:8000/health      # {"status":"healthy"}
```

Interactive API docs are available at `http://localhost:8000/docs`.

### 2. Frontend (Streamlit)

In a second terminal, with the virtual environment active:

```bash
streamlit run frontend/app.py
```

Streamlit serves the UI (by default on `http://localhost:8501`). If the backend
runs elsewhere, set `BACKEND_URL` before starting Streamlit. The sidebar shows
the backend URL and a connection indicator ("Connected" / "Backend not
reachable") with a **Check connection** button.

### 3. Using the UI

1. **Upload a paper** — choose a `.pdf` file in "1. Upload a paper".
2. **Ingest and index** — click **Ingest and index this paper**. This calls
   `/ingest` for a page/chunk preview, then `/index` to embed and persist. The
   sidebar "Active paper" panel updates immediately with the source, page count,
   chunk count and vector count.
3. **Ask a question** — type into the question box (it shows a red outline while
   empty/whitespace-only and a green outline once it contains text) and click
   **Ask**.
4. **View the answer** — a grounded answer is shown in green; an abstention is
   shown as a warning ("Not enough evidence…") rather than an error.
5. **View citations and evidence** — a **Sources** list shows the supporting
   pages/sources, and the **Retrieved Evidence** expander shows both the reranked
   evidence and the raw FAISS candidates with their scores.

Only one paper is active in the UI at a time: indexing a new PDF replaces the
previous index (the client sends `reset=true`). Use **Clear paper** to reset the
UI state.

---

## API reference

Base URL in local development: `http://localhost:8000`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Root probe — returns a short message. |
| `GET` | `/health` | Health check — `{"status":"healthy"}`. |
| `POST` | `/ingest` | Validate a PDF and return its page-aware chunks (stateless preview). |
| `POST` | `/index` | Embed the PDF's chunks and persist the FAISS index (append or replace). |
| `POST` | `/search` | Semantic retrieval only (no reranking, no LLM). |
| `POST` | `/ask` | Grounded answer with citations and both evidence orderings. |

### POST /ingest

`multipart/form-data` with a `file` field. Optional query params `chunk_size`
(default `1000`) and `chunk_overlap` (default `200`).

Response `200`:

```json
{
  "status": "success",
  "source": "paper.pdf",
  "num_pages": 2,
  "num_chunks": 3,
  "page_numbers": [1, 2],
  "chunks": [
    { "chunk_index": 0, "page_number": 1, "source": "paper.pdf", "text": "..." }
  ]
}
```

`415` for non-PDF uploads; `422` for empty/corrupt/text-free PDFs or invalid
chunk parameters. `page_number` is 1-based; pages without extractable text are
skipped, so `page_numbers` may be non-contiguous.

### POST /index

`multipart/form-data` with a `file` field. Optional query params `chunk_size`,
`chunk_overlap`, and `reset` (default `false`; `true` replaces the existing
index).

Response `200`:

```json
{
  "status": "success",
  "source": "paper.pdf",
  "num_pages": 12,
  "num_chunks": 70,
  "num_indexed": 70,
  "total_vectors": 70,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "manifest": { "model_name": "...", "embedding_dim": 384, "num_vectors": 70,
                "normalized": true, "chunk_size": 1000, "chunk_overlap": 200,
                "schema_version": 1, "created_at": "..." }
}
```

`num_indexed` is the number of chunks added in this call; `total_vectors` is the
index size afterwards. `415`/`422`/`500` follow the same rules as `/ingest`.

### POST /search

`application/json` body: `{ "query": "...", "top_k": 5 }` (`top_k >= 1`).
Returns `{ "query", "top_k", "num_results", "results": [...] }` where each result
is `{ chunk_index, page_number, source, text, score }` ordered by descending
cosine score. `404` if no index exists; `422` for a blank query or invalid
`top_k`. Not used by the Streamlit client.

### POST /ask

`application/json` body:

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `query` | string | — | Required, non-blank. |
| `candidates` | int | `20` | `1..100` (FAISS candidates before reranking) |
| `top_n` | int | `5` | `1..20`, and `top_n <= candidates` (evidence kept) |

Response `200`:

```json
{
  "query": "...",
  "answer": "...",
  "grounded": true,
  "llm_model": "gemini-3.8-flash",
  "num_candidates": 20,
  "num_evidence": 5,
  "citations": [
    { "marker": 1, "source": "resnet.pdf", "page_number": 2,
      "chunk_index": 9, "rerank_score": -1.98 }
  ],
  "retrieved": [
    { "chunk_index": 9, "page_number": 2, "source": "resnet.pdf",
      "text": "...", "score": 0.6964, "retrieval_rank": 0 }
  ],
  "reranked": [
    { "chunk_index": 23, "page_number": 4, "source": "resnet.pdf",
      "text": "...", "score": 0.5904, "retrieval_rank": 2,
      "rerank_rank": 0, "rerank_score": 0.4283 }
  ]
}
```

Notes:

* `retrieval_rank` and `rerank_rank` are **0-based**.
* `retrieved[]` carries the FAISS cosine `score`; `reranked[]` adds the
  cross-encoder `rerank_score` (a logit — higher is more relevant).
* The internal prompt is never returned.

Status codes: `422` (blank query, out-of-range limits, `top_n > candidates`,
malformed body), `404` (no index), `503` (missing `GEMINI_API_KEY` or config
error), `502` (model call failed), `500` (unreadable/inconsistent index).

A full endpoint reference with more detail is in
[`docs/api_reference.md`](docs/api_reference.md).

---

## Grounding and abstention

The system is built to avoid answers unsupported by the indexed paper.

* **Zero evidence:** if reranking produces no evidence, the LLM is **never
  called**. The service returns a fixed "no matching content" answer,
  `grounded: false`, and empty citations.
* **Insufficient evidence:** the grounded prompt instructs the model to emit the
  sentinel `INSUFFICIENT_EVIDENCE` on its **first line** when the evidence cannot
  answer the question. If it does, the result is `grounded: false` with empty
  citations. Only the first line is checked, so a normal answer that happens to
  contain the phrase is not misread.
* **Grounded:** otherwise the answer is returned with `grounded: true` and a
  citation for each evidence chunk.

An abstention is a **normal `200` response**, not an error, and the retrieved
and reranked evidence is still returned so the user can inspect what was
considered. In the UI, a grounded answer is shown in green; an abstention is
shown as a warning ("Not enough evidence…") with an explicit "This is an
abstention, not an error." caption.

---

## Citations

Citations are built in `backend/rag/citations.py` **from the reranked chunk
metadata**, not from the model's answer text:

* one citation per evidence chunk;
* `marker` = the evidence's `rerank_rank + 1` — the same bracketed number the
  prompt gives the model for that chunk;
* `source`, `page_number`, `chunk_index`, and `rerank_score` come straight from
  the retrieved chunk.

Because the model's prose is never parsed for citations, a citation can only
reference a page that was actually retrieved and passed as evidence. The
frontend renders these as the **Sources** list and the **Retrieved Evidence**
tabs (reranked evidence and raw FAISS candidates).

---

## Evaluation

The evaluation dataset is `evaluation/questions.json`: **15 locked questions**
(**12 answerable**, **3 unanswerable**) over the ResNet paper, each with a
`category`, `reference_answer`, `expected_pages`, and an `answerable` flag. It is
validated eagerly by the harness and opened read-only; it is **never modified**.

`evaluation/run_evaluation.py` runs the **real** pipeline (pdfplumber ingestion,
MiniLM embeddings, FAISS retrieval, the ms-marco cross-encoder, and Gemini) for
every question, indexing the paper into a scratch store
(`data/faiss_store/evaluation_tmp`) so evaluation runs never disturb the API's own
index.

```bash
# Real pipeline + Gemini generation (needs GEMINI_API_KEY)
python evaluation/run_evaluation.py

# Offline mode: generation stubbed, retrieval & reranking still real (no API key)
python evaluation/run_evaluation.py --skip-api

# Also run the RAGAS layer afterwards (needs GEMINI_API_KEY)
python evaluation/run_evaluation.py --ragas
```

Outputs:

* `evaluation/results/evaluation.json` — per-question traces plus a summary.
* `evaluation/results/ragas.jsonl` — per-query RAGAS metrics (only with `--ragas`),
  merged back into `evaluation.json` as a `ragas` block.

**Scoring dimensions**

* **Retrieval** (answerable questions): recall of the expected pages within the
  *reranked* evidence, and hit@1 (the top reranked chunk sits on an expected page).
* **Abstention** (unanswerable questions): did the pipeline take its own
  not-grounded path instead of producing a "grounded" answer?
* **Answer overlap**: deterministic content-word recall against the reference
  answer (threshold `0.5`) — a heuristic, not a substitute for reading answers.

**Committed results (real, from `evaluation/results/evaluation.json`)**

The repository contains one committed run, produced with `--skip-api`
(`llm_model: "offline-stub"`, note: *"Generation stubbed offline; retrieval/reranking
are real."*). Its real retrieval/abstention numbers:

| Metric | Value |
| --- | --- |
| Questions run | 15 (12 answerable / 3 unanswerable) |
| Retrieval top-1 hit | 6 / 12 |
| Retrieval any hit | 11 / 12 |
| Retrieval macro recall | 0.875 |
| Correct abstention | 3 / 3 |
| `answer_correct_count` | 0 / 12 — **not meaningful** (generation was stubbed) |

Because that run stubbed generation, its answer-quality numbers must not be read
as a measure of the model's answer quality.

---

## RAGAS

`evaluation/ragas_eval.py` is an optional layer over already-produced results. It
scores three metrics per query:

| Metric | Question it answers |
| --- | --- |
| `faithfulness` | Is the generated answer supported by the retrieved evidence? |
| `answer_relevancy` | Does the answer actually address the question? |
| `context_precision` | Is the retrieved evidence relevant and well ranked? |

Implementation notes:

* The judge LLM is the **same Gemini model** the pipeline uses (built from
  `GEMINI_API_KEY` / `GEMINI_MODEL`).
* `answer_relevancy` needs embeddings, so the project's own MiniLM embedder is
  adapted to RAGAS's interface — no extra embedding model is introduced.
* `context_precision` uses the reference-based variant for the 12 answerable
  questions and the reference-free variant for the 3 unanswerable ones.
* The results must be **really generated**: RAGAS **refuses to score** a
  `--skip-api` run (it raises rather than emitting meaningless metrics).
* Per-query rows are written to `evaluation/results/ragas.jsonl`, and aggregates
  are merged into `evaluation/results/evaluation.json`.

**No RAGAS results are committed to this repository** (there is no `ragas.jsonl`
or `ragas` block), so no RAGAS scores are reported here. To produce them, run:

```bash
python evaluation/run_evaluation.py --ragas   # real pipeline + RAGAS (needs a Gemini key)
# or, over an existing real (non-stub) evaluation.json:
python evaluation/ragas_eval.py
```

---

## Testing

The suite uses `pytest` (16 `test_*.py` modules plus shared fixtures/fakes,
covering ingestion, chunking, embeddings, FAISS, reranking, prompting, the RAG
service, the API, and the frontend).
`pytest.ini` registers `slow` (real model weights) and `live` (real Gemini) markers.

```bash
# Everything that does not call the live Gemini API
pytest -q -m "not live"
```

**Latest verified result:**

```
257 passed, 1 deselected, 1 warning in 32.04s
```

The single deselected test is the opt-in live test
(`tests/test_ask_api.py::test_live_ask_end_to_end`). It is marked both `live`
and `slow`, loads the **real** embedder, cross-encoder and Gemini, and is
**skipped unless `GEMINI_API_KEY` is set** in the process environment; it
consumes real Gemini quota, so it is not run by default. Total collection is
**258 tests** (257 non-live + 1 live).

Some non-live tests are marked `slow` because they load the real
sentence-transformers weights; they are included in the 257 above and require the
models to be available locally.

---

## Validation example

Manual end-to-end validation was performed in the Streamlit UI with the ResNet
paper, *Deep Residual Learning for Image Recognition*
(`data/raw_pdfs/resnet.pdf`, 12 pages):

| Step | Observed |
| --- | --- |
| Upload | `resnet.pdf` selected |
| Ingest + index | 12 pages → 70 chunks → 70 vectors (`sentence-transformers/all-MiniLM-L6-v2`) |
| Sidebar | "Active paper" panel updated to the indexed paper immediately |
| Ask (answerable) | Real Gemini answer returned, grounded, with page/source citations |
| Ask (unanswerable) | A question about inference latency produced an **abstention** instead of a guess |

The paper's page count (12) was verified from the PDF, and the 70-vector index is
reflected in the persisted store manifest. Retrieval defaults (20 candidates
reranked to 5 evidence chunks) match the pipeline configuration. Per-question
real retrieval/reranking evidence for the answerable ResNet question is recorded
in [`docs/evidence/rerank_before_after.json`](docs/evidence/rerank_before_after.json)
and its companion PNG.

---

## Known limitations

* **Text-based PDFs only.** Scanned/image-only PDFs have no extractable text and
  are rejected with `422`. There is no OCR.
* **One active paper in the UI.** Indexing a new PDF in Streamlit replaces the
  previous index (`reset=true`). The API itself can append multiple documents
  when `reset=false`.
* **Exact, brute-force search.** The index is `IndexFlatIP`, which compares the
  query against every vector — accurate but linearly scaling; it is not intended
  for very large corpora.
* **Page-aware chunking constraint.** Chunks never span page boundaries. This
  keeps citations exact but means a single chunk cannot contain context that
  crosses a page break.
* **Model/parameter coupling to a persisted index.** The store records the
  embedding model and chunk parameters; loading it with a mismatched embedding
  model (or an unknown newer schema) raises an index-load error.
* **Gemini dependency.** `/ask` needs `GEMINI_API_KEY` and depends on the
  external Gemini API and its quota/availability. Retrieval and reranking run
  fully locally and offline.
* **First-call latency.** The embedding and cross-encoder models (and, on first
  ever use, their weights) load lazily, so the first index/ask can be slower.
* **No query routing, decomposition, hybrid/BM25 search, conversation memory,
  authentication, or streaming.** Retrieval is pure dense FAISS over the query
  as written.

---

## Security and secrets

* `GEMINI_API_KEY` is supplied through environment configuration (or a
  gitignored `.env`) and is **never** committed, logged, defaulted, or included
  in error messages; it is excluded from the settings `repr`.
* `.env` is listed in `.gitignore`; `.env.example` ships with an **empty** key.
* The Gemini key is used **only** in the backend. The Streamlit frontend never
  receives or exposes it — it only talks to the backend HTTP API.
* Repositories should never contain real keys; rotate any key that is
  accidentally committed.

---

## Further documentation

* [`docs/architecture.md`](docs/architecture.md) — detailed architecture and
  component responsibilities.
* [`docs/api_reference.md`](docs/api_reference.md) — endpoint-by-endpoint
  reference.
* [`docs/evidence/rerank_before_after.json`](docs/evidence/rerank_before_after.json)
  and [`rerank_before_after.png`](docs/evidence/rerank_before_after.png) — real
  FAISS-retrieval vs. cross-encoder ordering for one committed evaluation query.
