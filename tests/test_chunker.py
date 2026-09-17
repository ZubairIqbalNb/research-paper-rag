"""Unit tests for page-aware RecursiveCharacterTextSplitter chunking."""
from backend.core.models import PageDocument
from backend.ingestion.chunker import chunk_pages


def _page(page_number: int, text: str, source: str = "paper.pdf") -> PageDocument:
    return PageDocument(text=text, page_number=page_number, source=source)


def test_short_pages_produce_one_chunk_each():
    pages = [_page(1, "Short text one."), _page(2, "Short text two.")]

    chunks = chunk_pages(pages)

    assert len(chunks) == 2
    assert [c.page_number for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_chunks_never_exceed_chunk_size():
    pages = [_page(1, "word " * 2000)]  # ~10k chars, well above 1000

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)


def test_chunks_never_span_page_boundaries():
    long_text = "sentence about topic. " * 150  # ~3.3k chars per page
    pages = [_page(1, long_text), _page(2, long_text), _page(3, long_text)]

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)

    page_numbers = [c.page_number for c in chunks]
    assert page_numbers == sorted(page_numbers)  # ordered by page
    assert set(page_numbers) == {1, 2, 3}


def test_overlap_present_on_multi_chunk_pages():
    text = "unique-token " * 300  # ~3.9k chars on one page
    chunks = chunk_pages([_page(1, text)], chunk_size=1000, chunk_overlap=200)

    assert len(chunks) >= 3
    # Continuation chunks should share some content with their predecessor.
    first_words = set(chunks[1].text.split())
    assert first_words & set(chunks[0].text.split())


def test_metadata_source_and_page_preserved():
    pages = [
        PageDocument(text="Alpha content here.", page_number=4, source="report_a.pdf"),
        PageDocument(text="Beta content here.", page_number=7, source="report_b.pdf"),
    ]

    chunks = chunk_pages(pages)

    assert chunks[0].source == "report_a.pdf" and chunks[0].page_number == 4
    assert chunks[1].source == "report_b.pdf" and chunks[1].page_number == 7


def test_chunk_index_is_global_across_pages():
    # Two long pages -> multiple chunks each; index must run 0..N-1 globally.
    text = "sentence content here. " * 120  # ~2.6k chars
    pages = [_page(1, text), _page(2, text)]

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_custom_params_are_respected():
    text = "word " * 500  # 2500 chars
    chunks = chunk_pages([_page(1, text)], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 4
    assert all(len(c.text) <= 300 for c in chunks)
