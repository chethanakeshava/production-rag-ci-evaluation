
from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Split page text into overlapping word-based chunks.

    Each chunk preserves its source filename and page number.
    Chunk size and overlap are measured in words in this initial version.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                continue

            chunks.append(
                {
                    "text": " ".join(chunk_words),
                    "source": page["source"],
                    "page": page["page"],
                }
            )

            if start + chunk_size >= len(words):
                break

    return chunks
