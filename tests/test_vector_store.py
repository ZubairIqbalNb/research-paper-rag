"""FAISS vector store tests: build, persistence, search, metadata mapping, top_k."""
import json

import faiss
import numpy as np
import pytest
from fakes import DeterministicEmbedder

from backend.core.models import Chunk
from backend.retrieval.vector_store import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    SCHEMA_VERSION,
    FaissVectorStore,
    IndexLoadError,
    IndexNotBuiltError,
)

# Disjoint vocabulary across sources/pages so ranking expectations are stable.
CHUNK_SPECS = [
    ("retrieval evaluation metrics measure ranking quality of search systems", "paper_a.pdf", 1),
    ("self attention transformers process token sequences in parallel", "paper_a.pdf", 2),
    ("photosynthesis converts sunlight into chemical energy in plants", "paper_b.pdf", 7),
]
QUERY = "retrieval evaluation metrics ranking"


def _store(chunks, embedder, *, chunk_size=1000, chunk_overlap=200):
    """Build a store the way the retrieval service does."""
    vectors = embedder.embed([chunk.text for chunk in chunks])
    return FaissVectorStore.build(
        chunks,
        vectors,
        model_name=embedder.model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _tamper_metadata(store_dir, mutate) -> None:
    path = store_dir / METADATA_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _as_tuples(results):
    return [
        (r.chunk_index, r.page_number, r.source, r.text, round(r.score, 6)) for r in results
    ]


# ------------------------------------------------------------------ index build
def test_build_indexes_every_chunk_with_embedder_dimension(make_chunks, deterministic_embedder):
    chunks = make_chunks(CHUNK_SPECS)

    store = _store(chunks, deterministic_embedder)

    assert store.size == len(chunks)
    assert store.manifest.embedding_dim == deterministic_embedder.dimension
    assert store.manifest.num_vectors == len(chunks)


def test_build_uses_dimension_from_embeddings_not_hardcoded(make_chunks):
    chunks = make_chunks(CHUNK_SPECS)

    store = _store(chunks, DeterministicEmbedder(dimension=32))

    assert store.manifest.embedding_dim == 32


def test_manifest_records_model_and_chunking_params(make_chunks, deterministic_embedder):
    chunks = make_chunks(CHUNK_SPECS)

    manifest = _store(chunks, deterministic_embedder, chunk_size=800, chunk_overlap=150).manifest

    assert manifest.model_name == deterministic_embedder.model_name
    assert manifest.chunk_size == 800
    assert manifest.chunk_overlap == 150
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.normalized is True
    assert manifest.created_at  # ISO-8601 UTC timestamp


def test_build_rejects_empty_chunks(deterministic_embedder):
    with pytest.raises(ValueError, match="empty chunk list"):
        FaissVectorStore.build(
            [], deterministic_embedder.embed(["x"]), model_name="stub"
        )


def test_build_rejects_vector_count_mismatch(make_chunks, deterministic_embedder):
    chunks = make_chunks(CHUNK_SPECS)
    vectors = deterministic_embedder.embed([c.text for c in chunks[:2]])

    with pytest.raises(ValueError, match="vectors for"):
        FaissVectorStore.build(chunks, vectors, model_name="stub")


# ------------------------------------------------------------------ persistence
def test_save_writes_index_and_metadata_separately(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    chunks = make_chunks(CHUNK_SPECS)
    store = _store(chunks, deterministic_embedder)

    store.save(tmp_store_dir)

    index_path = tmp_store_dir / INDEX_FILENAME
    metadata_path = tmp_store_dir / METADATA_FILENAME
    assert index_path.is_file() and metadata_path.is_file()
    # Vectors live in the binary index; metadata is plain JSON (no vectors).
    assert index_path.read_bytes()[:4] != b"json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert set(payload) == {"manifest", "chunks"}
    assert [row for row in payload["chunks"]] == [
        {
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "source": c.source,
            "text": c.text,
        }
        for c in chunks
    ]


def test_persisted_index_is_exact_inner_product(make_chunks, deterministic_embedder, tmp_store_dir):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)
    store.save(tmp_store_dir)

    index = faiss.read_index(str(tmp_store_dir / INDEX_FILENAME))

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.d == deterministic_embedder.dimension
    assert index.ntotal == len(CHUNK_SPECS)


def test_save_then_load_preserves_search_results(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    chunks = make_chunks(CHUNK_SPECS)
    store = _store(chunks, deterministic_embedder)
    store.save(tmp_store_dir)
    query_vector = deterministic_embedder.embed([QUERY])[0]

    before = store.search(query_vector, top_k=3)
    reloaded = FaissVectorStore.load(tmp_store_dir)
    after = reloaded.search(query_vector, top_k=3)

    assert _as_tuples(after) == _as_tuples(before)
    assert reloaded.size == store.size
    assert reloaded.manifest.embedding_dim == store.manifest.embedding_dim


def test_loaded_metadata_rows_keep_chunk_order(make_chunks, deterministic_embedder, tmp_store_dir):
    chunks = make_chunks(CHUNK_SPECS)
    _store(chunks, deterministic_embedder).save(tmp_store_dir)

    reloaded = FaissVectorStore.load(tmp_store_dir)

    assert [(c.source, c.page_number, c.chunk_index, c.text) for c in reloaded.chunks] == [
        (c.source, c.page_number, c.chunk_index, c.text) for c in chunks
    ]


def test_load_missing_store_raises_index_not_built(tmp_store_dir):
    with pytest.raises(IndexNotBuiltError):
        FaissVectorStore.load(tmp_store_dir)


def test_load_detects_vector_count_mismatch(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    _store(make_chunks(CHUNK_SPECS), deterministic_embedder).save(tmp_store_dir)
    _tamper_metadata(tmp_store_dir, lambda p: p["manifest"].__setitem__("num_vectors", 99))

    with pytest.raises(IndexLoadError, match="inconsistent"):
        FaissVectorStore.load(tmp_store_dir)


def test_load_detects_missing_metadata_rows(make_chunks, deterministic_embedder, tmp_store_dir):
    _store(make_chunks(CHUNK_SPECS), deterministic_embedder).save(tmp_store_dir)
    _tamper_metadata(tmp_store_dir, lambda p: p["chunks"].pop())

    with pytest.raises(IndexLoadError, match="inconsistent"):
        FaissVectorStore.load(tmp_store_dir)


def test_load_detects_dimension_mismatch(make_chunks, deterministic_embedder, tmp_store_dir):
    _store(make_chunks(CHUNK_SPECS), deterministic_embedder).save(tmp_store_dir)
    _tamper_metadata(tmp_store_dir, lambda p: p["manifest"].__setitem__("embedding_dim", 999))

    with pytest.raises(IndexLoadError, match="Dimension mismatch"):
        FaissVectorStore.load(tmp_store_dir)


def test_load_detects_embedding_model_mismatch(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    _store(make_chunks(CHUNK_SPECS), deterministic_embedder).save(tmp_store_dir)

    with pytest.raises(IndexLoadError, match="embedding model"):
        FaissVectorStore.load(tmp_store_dir, expect_model="some-other-model")


def test_load_rejects_corrupt_metadata(make_chunks, deterministic_embedder, tmp_store_dir):
    _store(make_chunks(CHUNK_SPECS), deterministic_embedder).save(tmp_store_dir)
    (tmp_store_dir / METADATA_FILENAME).write_text("{ not json", encoding="utf-8")

    with pytest.raises(IndexLoadError, match="Could not read index"):
        FaissVectorStore.load(tmp_store_dir)


# ---------------------------------------------------------------------- search
def test_search_ranks_the_closest_chunk_first(make_chunks, deterministic_embedder):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)

    results = store.search(deterministic_embedder.embed([QUERY])[0], top_k=3)

    assert results[0].source == "paper_a.pdf" and results[0].page_number == 1
    assert "retrieval evaluation metrics" in results[0].text
    assert results[0].score >= results[1].score >= results[2].score
    assert isinstance(results[0].score, float)


def test_search_maps_metadata_back_to_original_chunks(make_chunks, deterministic_embedder):
    chunks = make_chunks(CHUNK_SPECS)
    store = _store(chunks, deterministic_embedder)
    by_key = {(c.source, c.page_number, c.chunk_index): c.text for c in chunks}

    results = store.search(deterministic_embedder.embed([QUERY])[0], top_k=len(chunks))

    assert len(results) == len(chunks)
    for result in results:
        assert by_key[(result.source, result.page_number, result.chunk_index)] == result.text
    # Every FAISS row is reported exactly once, so no result is cross-wired.
    assert sorted(r.chunk_index for r in results) == sorted(c.chunk_index for c in chunks)


def test_search_respects_top_k(make_chunks, deterministic_embedder):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)

    results = store.search(deterministic_embedder.embed([QUERY])[0], top_k=1)

    assert len(results) == 1


def test_search_clamps_top_k_above_index_size(make_chunks, deterministic_embedder):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)

    results = store.search(deterministic_embedder.embed([QUERY])[0], top_k=99)

    # FAISS pads with -1 rows when asked for more neighbours than exist.
    assert len(results) == store.size == len(CHUNK_SPECS)


def test_search_rejects_non_positive_top_k(make_chunks, deterministic_embedder):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)

    with pytest.raises(ValueError, match="top_k"):
        store.search(deterministic_embedder.embed([QUERY])[0], top_k=0)


def test_search_rejects_query_dimension_mismatch(make_chunks, deterministic_embedder):
    store = _store(make_chunks(CHUNK_SPECS), deterministic_embedder)

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search(np.zeros(7, dtype=np.float32), top_k=1)


# -------------------------------------------------------------- reproducibility
def test_index_is_reproducible_from_the_same_chunks(make_chunks, tmp_store_dir):
    chunks = make_chunks(CHUNK_SPECS)
    first_embedder = DeterministicEmbedder()
    second_embedder = DeterministicEmbedder()

    first_vectors = first_embedder.embed([c.text for c in chunks])
    second_vectors = second_embedder.embed([c.text for c in chunks])
    np.testing.assert_array_equal(first_vectors, second_vectors)

    first = _store(chunks, first_embedder)
    second = _store(chunks, second_embedder)
    first.save(tmp_store_dir / "one")
    second.save(tmp_store_dir / "two")

    query = first_embedder.embed([QUERY])[0]
    assert _as_tuples(first.search(query, top_k=3)) == _as_tuples(second.search(query, top_k=3))

    first_payload = json.loads((tmp_store_dir / "one" / METADATA_FILENAME).read_text())
    second_payload = json.loads((tmp_store_dir / "two" / METADATA_FILENAME).read_text())
    assert first_payload["chunks"] == second_payload["chunks"]


def test_append_extends_rows_without_disturbing_existing_ones(make_chunks, deterministic_embedder):
    first_batch = make_chunks(CHUNK_SPECS[:2])
    store = _store(first_batch, deterministic_embedder)
    extra = [
        Chunk(
            text="contrastive pretraining objectives for dense retrieval",
            chunk_index=2,
            page_number=3,
            source="paper_c.pdf",
        )
    ]

    manifest = store.add(
        extra,
        deterministic_embedder.embed([c.text for c in extra]),
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert store.size == 3
    assert manifest.num_vectors == 3
    results = store.search(
        deterministic_embedder.embed(["dense retrieval pretraining objectives"])[0], top_k=1
    )
    assert results[0].source == "paper_c.pdf" and results[0].page_number == 3


def test_append_with_different_chunk_params_marks_them_unknown(
    make_chunks, deterministic_embedder
):
    chunks = make_chunks(CHUNK_SPECS)
    store = _store(chunks, deterministic_embedder, chunk_size=1000, chunk_overlap=200)

    manifest = store.add(
        chunks,
        deterministic_embedder.embed([c.text for c in chunks]),
        chunk_size=400,
        chunk_overlap=50,
    )

    assert manifest.chunk_size is None and manifest.chunk_overlap is None
