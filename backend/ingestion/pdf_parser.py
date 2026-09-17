"""PDF text extraction, page by page, using pdfplumber.

Knows nothing about chunking or HTTP; later phases can reuse it as-is.
"""
from pathlib import Path

import pdfplumber

from backend.core.models import PageDocument


def extract_pages(pdf_path: Path, source_label: str | None = None) -> list[PageDocument]:
    """Extract text from each page of a PDF as 1-based PageDocuments.

    Args:
        pdf_path: Path to the PDF file.
        source_label: Metadata ``source`` recorded on every page/chunk.
            Defaults to the file's own name; callers (e.g. the upload API)
            can pass the user-facing filename instead.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        ValueError: If the file is not a readable PDF or yields no text
            at all (e.g. a scanned image-only PDF).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source = source_label if source_label is not None else pdf_path.name

    pages: list[PageDocument] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for zero_based_index, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue  # skip blank pages (e.g. covers, image-only)
                pages.append(
                    PageDocument(
                        text=text,
                        page_number=zero_based_index + 1,  # 1-based for citations
                        source=source,
                    )
                )
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{pdf_path.name}': {exc}") from exc

    if not pages:
        raise ValueError(
            f"No extractable text found in '{pdf_path.name}'. "
            "The PDF may be scanned/image-only (OCR is not supported)."
        )
    return pages
