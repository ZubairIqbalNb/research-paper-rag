"""RetrievalService tests: chunk -> embedding -> persisted index -> search."""
import json

import pytest

from backend.retrieval.service import RetrievalService
from backend.retrieval.vector_store import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    IndexLoadError,
    IndexNotBuiltError,
)

CHUNK_SPECS = [
    ("retrieval evaluation metrics measure ranking quality of search systems", "paper_a.pdf", 1),
    ("self attention transformers process token sequences in parallel", "paper_a.pdf", 2),
    ("photosynthesis converts sunlight into chemical energy in plants", "paper_b.pdf", 7),
]
QUERY = "retrieval evaluation metrics ranking"


def _service(tmp_store_dir, embedder):
    return RetrievalService(store_dir=tmp_store_dir, embedder=embedder)


def test_add_chunks_persists_index_and_metadata(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    service = _service(tmp_store_dir, deterministic_embedder)

    manifest = service.add_chunks(make_chunks(CHUNK_SPECS))

    assert manifest.num_vectors == len(CHUNK_SPECS)
    assert manifest.model_name == deterministic_embedder.model_name
    assert (tmp_store_dir / INDEX_FILENAME).is_file()
    assert (tmp_store_dir / METADATA_FILENAME).is_file()


def test_indexed_chunks_are_searchable_after_restart(
    make_chunks, deterministic_embedder, tmp_store_dir
):
    _service(tmp_store_dir, deterministic_embedder).add_chunks(make_chunks(CHUNK_SPECS))

    # Fresh service: nothing in memory, everything reloaded from disk.
    restarted = _service(tmp_store_dir, deterministic_embedder)
    results = restarted.search(QUERY, top_k=2)

    assert results[0].source == "paper_a.pdf"
    assert results[0].page_number == 1
    assert "retrieval evaluation metrics" in results[0].text
    assert results[0].chunk_index == 0


def test_search_preserves_phase1_metadata(make_chunks, deterministic_embedder, tmp_store_dir):
    chunks = make_chunks(CHUNK_SPECS)
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(chunks)

    results = service.search(QUERY, top_k=3)

    by_key = {(c.source, c.page_number, c.chunk_index): c.text for c in chunks}
    for result in results:
        assert by_key[(result.source, result.page_number, result.chunk_index)] == result.text


def test_add_chunks_accumulates_documents(make_chunks, deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(make_chunks(CHUNK_SPECS[:2]))
    manifest = service.add_chunks(make_chunks(CHUNK_SPECS[2:]))

    assert manifest.num_vectors == 3
    assert json.loads((tmp_store_dir / METADATA_FILENAME).read_text())["manifest"][
        "num_vectors"
    ] == 3


def test_reset_replaces_the_existing_index(make_chunks, deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(make_chunks(CHUNK_SPECS))

    manifest = service.add_chunks(make_chunks(CHUNK_SPECS[:1]), reset=True)

    assert manifest.num_vectors == 1
    results = service.search(QUERY, top_k=5)
    assert len(results) == 1


def test_add_chunks_rejects_empty_list(deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)

    with pytest.raises(ValueError, match="empty chunk list"):
        service.add_chunks([])


def test_search_without_an_index_raises_index_not_built(deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)

    with pytest.raises(IndexNotBuiltError, match="No index found"):
        service.search(QUERY, top_k=1)


def test_search_rejects_blank_query(make_chunks, deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(make_chunks(CHUNK_SPECS))

    with pytest.raises(ValueError, match="non-empty"):
        service.search("   ", top_k=1)


def test_search_rejects_non_positive_top_k(make_chunks, deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(make_chunks(CHUNK_SPECS))

    with pytest.raises(ValueError, match="top_k"):
        service.search(QUERY, top_k=0)


def test_search_clamps_top_k_above_index_size(make_chunks, deterministic_embedder, tmp_store_dir):
    service = _service(tmp_store_dir, deterministic_embedder)
    service.add_chunks(make_chunks(CHUNK_SPECS))

    assert len(service.search(QUERY, top_k=50)) == len(CHUNK_SPECS)


def test_loading_an_index_built_by_another_model_is_rejected(
    make_chunks, tmp_store_dir
):
    from fakes import DeterministicEmbedder

    built_with = DeterministicEmbedder(model_name="model-a")
    _service(tmp_store_dir, built_with).add_chunks(make_chunks(CHUNK_SPECS))

    other = _service(tmp_store_dir, DeterministicEmbedder(model_name="model-b"))

    with pytest.raises(IndexLoadError, match="embedding model"):
        other.search(QUERY, top_k=1)
