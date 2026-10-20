
from pathlib import Path

import pymupdf


def load_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF, preserving source and page metadata.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A list of dictionaries, one per page, containing:
        - text
        - source
        - page

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF or contains no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "text": text,
                        "source": path.name,
                        "page": page_number,
                    }
                )

    if not pages:
        raise ValueError(
            f"No extractable text found in PDF: {path}. "
            "It may be empty or scanned."
        )

    return pages
