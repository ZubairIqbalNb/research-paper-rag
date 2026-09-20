# API Reference

The backend is a FastAPI application defined in `health.py` (module-level
`app`). Routes are grouped in `backend/api/`: `ingest.py` (`/ingest`),
`retrieval.py` (`/index`, `/search`) and `ask.py` (`/ask`).

* **Default base URL (client):** `http://localhost:8000` — the Streamlit client
  reads `BACKEND_URL` (see `frontend/api_client.py`) and trims any trailing slash.
* **Uploads:** `multipart/form-data` with a single `file` field.
* **JSON endpoints:** `application/json`.
* **Errors:** FastAPI's standard `{"detail": ...}` body. The `detail` is either a
  string (raised by the app) or FastAPI's validation-error list (422). The
  frontend maps the common status codes to user-safe messages
  (`frontend/api_client.py`).

Endpoints implemented: `GET /`, `GET /health`, `POST /ingest`, `POST /index`,
`POST /search`, `POST /ask`.

Two response conventions matter throughout:

* **`retrieval_rank` and `rerank_rank` are 0-based** (Python `enumerate` indices).
* **The grounded prompt is internal.** It is stored on the result for
  testing/debugging but is deliberately **not** exposed in any response field.

Secrets (`GEMINI_API_KEY`) are never returned, logged, or included in errors.

---

## GET /

**Purpose:** Liveness/root probe. Not used by the frontend.

**Request:** none.

**Response 200:**

```json
{ "message": "API is running. Go to /docs or /health" }
```

---

## GET /health

**Purpose:** Backend health check; used by the Streamlit sidebar to show the
connection indicator.

**Request:** none.

**Response 200:**

```json
{ "status": "healthy" }
```

---

## POST /ingest

**Purpose:** Stateless PDF preview. Validates and extracts the uploaded PDF and
returns its **page-aware chunks**. Writes nothing to disk and does not index.

**Request:** `multipart/form-data`, field `file` (a PDF).

**Query parameters:**

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `chunk_size` | int | `1000` | Splitter chunk size. |
| `chunk_overlap` | int | `200` | Splitter overlap. |

**Response 200:**

```json
{
  "status": "success",
  "source": "paper.pdf",
  "num_pages": 2,
  "num_chunks": 3,
  "page_numbers": [1, 2],
  "chunks": [
    {
      "chunk_index": 0,
      "page_number": 1,
      "source": "paper.pdf",
      "text": "..."
    }
  ]
}
```

* `page_numbers` may be non-contiguous: pages with no extractable text are
  skipped. `page_number` is 1-based.
* `chunk_index` is a global ordinal across the document; chunk text never spans
  a page boundary.
* Client-facing fields: all of the above. The chunk list is the complete preview
  payload, not an internal artifact.

**Validation and status codes:**

| Status | When |
| --- | --- |
| `415` | Filename does not end in `.pdf` and the content type is not `application/pdf`. Detail: *"Unsupported file type: only PDFs are accepted"*. |
| `422` | Upload is empty or lacks the `%PDF-` header; the PDF is corrupt; or it has no extractable text (e.g. scanned/image-only). |
| `422` | Invalid chunk parameters (`chunk_size <= 0`, `chunk_overlap < 0`, or `chunk_size < chunk_overlap`). |
| `500` | Internal temp-file error (defensive). |

---

## POST /index

**Purpose:** Embed the uploaded PDF's chunks and persist them to the FAISS store
(append or replace). Uses the same PDF → pages → chunks pipeline as `/ingest`, so
indexed metadata equals what `/ingest` returns.

**Request:** `multipart/form-data`, field `file` (a PDF).

**Query parameters:**

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `chunk_size` | int | `1000` | Splitter chunk size. |
| `chunk_overlap` | int | `200` | Splitter overlap. |
| `reset` | bool | `false` | `true` replaces the existing index instead of appending. The Streamlit client always sends `reset=true`. |

**Response 200:**

```json
{
  "status": "success",
  "source": "paper.pdf",
  "num_pages": 12,
  "num_chunks": 70,
  "num_indexed": 70,
  "total_vectors": 70,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "manifest": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "num_vectors": 70,
    "normalized": true,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "schema_version": 1,
    "created_at": "2026-09-20T06:37:23+00:00"
  }
}
```

* `num_indexed` equals the number of chunks added in this call;
  `total_vectors` is the index size afterwards (so accumulator runs add up).
* `manifest` is the persisted index header (`IndexManifest.to_dict()`).
* Client-facing fields: all of the above.

**Validation and status codes:**

| Status | When |
| --- | --- |
| `415` | Non-PDF upload (same rule as `/ingest`). |
| `422` | Empty/corrupt/text-free PDF, invalid chunk params, or an empty chunk list. |
| `422` | Embedding/model error surfaced as `ValueError` (e.g. dimension mismatch on append). |
| `500` | The persisted index exists but is unreadable/inconsistent (`IndexLoadError`). |

---

## POST /search

**Purpose:** Semantic retrieval only — the chunks most similar to the query.
No reranking, no LLM. Exposed by the API but **not used by the Streamlit client**.

**Request:** `application/json`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query` | str | — | Required. Must be non-blank after stripping. |
| `top_k` | int | `5` | `>= 1`. If greater than the number of indexed vectors, results are clamped (no phantom rows). |

**Example request:**

```json
{ "query": "retrieval evaluation metrics ranking quality", "top_k": 5 }
```

**Response 200:**

```json
{
  "query": "retrieval evaluation metrics ranking quality",
  "top_k": 5,
  "num_results": 2,
  "results": [
    {
      "chunk_index": 0,
      "page_number": 1,
      "source": "paper.pdf",
      "text": "...",
      "score": 0.71
    }
  ]
}
```

* `score` is the FAISS cosine similarity (inner product on normalized vectors),
  ordered best first.
* `top_k` echoes the **requested** value; `num_results` is the actual returned
  count, which may be smaller (clamped to the index size).
* Client-facing fields: all of the above.

**Validation and status codes:**

| Status | When |
| --- | --- |
| `422` | Blank query, `top_k < 1`, or a missing/invalid body (FastAPI schema). |
| `404` | No index exists yet. Detail: *"No index found ..."*. |
| `500` | The persisted index is unreadable/inconsistent (`IndexLoadError`). |

---

## POST /ask

**Purpose:** Grounded question answering over the indexed paper. Retrieves
candidates from FAISS, reranks them with the cross-encoder, builds a grounded
prompt, and asks Gemini. Returns the answer, its grounding state, citations, and
**both** evidence orderings.

**Request:** `application/json`

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `query` | str | — | Required, non-blank. |
| `candidates` | int | `20` | `1 <= candidates <= 100` (FAISS candidates before reranking). |
| `top_n` | int | `5` | `1 <= top_n <= 20`, and `top_n <= candidates` (evidence kept). |

A model validator rejects requests where `top_n > candidates`.

**Example request:**

```json
{ "query": "What degradation problem do the authors identify when increasing the depth of plain neural networks?" }
```

**Response 200** (exact key set asserted by the API tests):

```json
{
  "query": "...",
  "answer": "Deeper plain networks... [1]",
  "grounded": true,
  "llm_model": "gemini-3.8-flash",
  "num_candidates": 20,
  "num_evidence": 5,
  "citations": [
    { "marker": 1, "source": "resnet.pdf", "page_number": 2, "chunk_index": 9, "rerank_score": -1.98 }
  ],
  "retrieved": [
    {
      "chunk_index": 9,
      "page_number": 2,
      "source": "resnet.pdf",
      "text": "...",
      "score": 0.6964,
      "retrieval_rank": 0
    }
  ],
  "reranked": [
    {
      "chunk_index": 23,
      "page_number": 4,
      "source": "resnet.pdf",
      "text": "...",
      "score": 0.5904,
      "retrieval_rank": 2,
      "rerank_rank": 0,
      "rerank_score": 0.4283
    }
  ]
}
```

Field conventions (all client-facing):

* `num_candidates` = length of `retrieved` (FAISS ordering);
  `num_evidence` = length of `reranked` (cross-encoder ordering).
* `retrieved[]` items carry the FAISS cosine `score` and a 0-based
  `retrieval_rank`; they do **not** carry `rerank_score`.
* `reranked[]` items preserve the cosine `score`, add the cross-encoder
  `rerank_score` (a logit; higher is more relevant), the 0-based
  `retrieval_rank` they came from, and the 0-based `rerank_rank`.
* `citations[]` are derived from reranked chunk metadata (`marker = rerank_rank + 1`),
  never parsed from the model's answer text.
* `grounded` is `false` for both abstention paths (zero evidence or the model
  signalling `INSUFFICIENT_EVIDENCE`); citations are empty in that case. An
  abstention is a normal `200`, not an error.
* The internal prompt is never returned.

**Status codes:**

| Status | When |
| --- | --- |
| `422` | Blank query; `candidates`/`top_n` out of range; `top_n > candidates`; missing/invalid body. Schema validation runs before the API-key check. |
| `404` | No index exists yet (`IndexNotBuiltError`). |
| `503` | `GEMINI_API_KEY` is not configured (checked before any model call). |
| `503` | A `ConfigurationError` surfaces during generation. |
| `502` | The model call failed or returned no usable text (`LLMError`). |
| `500` | The persisted index is unreadable/inconsistent (`IndexLoadError`). |

---

## Error mapping summary

| Status | Meaning |
| --- | --- |
| `415` | Unsupported upload type (not a PDF). |
| `422` | Request/validation error: bad chunk params, unreadable or text-free PDF, blank query, out-of-range limits, malformed body. |
| `404` | No paper indexed yet. |
| `500` | The backend could not read its search index. |
| `502` | Gemini could not produce an answer. |
| `503` | Missing Gemini API key configuration. |

## Examples caveat

The JSON above shows the **exact field names and types** produced by the code and
asserted by the tests. Where numeric values appear (scores, counts, hashes), they
are illustrative of the schema; the only fully real captured payload used in this
repository's documentation is the q01 run in
[`docs/evidence/rerank_before_after.json`](evidence/rerank_before_after.json),
which is derived from the committed `evaluation/results/evaluation.json`.
