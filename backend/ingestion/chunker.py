"""Page-aware chunking with LangChain's RecursiveCharacterTextSplitter.

Each page is chunked independently, so chunks never span page boundaries
and every chunk can be cited to exactly one page.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.models import Chunk, PageDocument

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_pages(
    pages: list[PageDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split pages into chunks, preserving page_number and source.

    Page-level chunking: since the splitter preserves order and never
    merges non-adjacent text, all chunks from one page inherit that
    page's number. chunk_index is a global ordinal across the document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )

    chunks: list[Chunk] = []
    for page in pages:
        for text in splitter.split_text(page.text):
            chunks.append(
                Chunk(
                    text=text,
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    source=page.source,
                )
            )
    return chunks
