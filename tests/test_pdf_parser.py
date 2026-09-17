"""Unit tests for pdfplumber page extraction."""
from pathlib import Path

import pytest

from backend.ingestion.pdf_parser import extract_pages


def test_extract_multi_page_pdf_preserves_order_and_numbering(make_pdf):
    pdf = make_pdf("paper.pdf", [
        "Alpha page. This is the first page of text.",
        "Beta page. This is the second page of text.",
        "Gamma page. This is the third page of text.",
    ])

    pages = extract_pages(pdf)

    assert [p.page_number for p in pages] == [1, 2, 3]  # 1-based
    assert all(p.source == "paper.pdf" for p in pages)
    assert "Alpha page" in pages[0].text
    assert "Beta page" in pages[1].text
    assert "Gamma page" in pages[2].text


def test_extract_empty_pages_are_skipped_but_numbering_stays_absolute(make_pdf):
    # Page 1 has text, page 2 is effectively blank, page 3 has text.
    pdf = make_pdf("gaps.pdf", [
        "First page with content.",
        "",
        "Third page with content.",
    ])

    pages = extract_pages(pdf)

    assert [p.page_number for p in pages] == [1, 3]
    assert len(pages) == 2


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "does_not_exist.pdf")


def test_non_pdf_file_raises_valueerror(tmp_path):
    fake = tmp_path / "fake.pdf"
    fake.write_text("this is not a pdf at all")

    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_pages(fake)


def test_no_extractable_text_raises_valueerror(tmp_path):
    # A structurally valid PDF with no text content (image-only style).
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf_path = tmp_path / "blank.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pages(pdf_path)


def test_source_is_filename_not_full_path(make_pdf):
    pdf = make_pdf("some_long_name.pdf", ["Just some text."])
    pages = extract_pages(pdf)
    assert pages[0].source == "some_long_name.pdf"
    assert str(Path(pdf)) not in pages[0].source
