
import pymupdf
import pytest

from app.ingestion.document_loader import load_pdf


def create_test_pdf(path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Production RAG retrieves relevant document evidence.",
        )
        document.save(path)


def test_load_pdf_extracts_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert "retrieves relevant document evidence" in pages[0]["text"]
    assert pages[0]["source"] == "sample.pdf"
    assert pages[0]["page"] == 1


def test_load_pdf_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf(text_file)


def test_load_pdf_rejects_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    with pytest.raises(ValueError, match="No extractable text"):
        load_pdf(pdf_path)
