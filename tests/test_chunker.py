
import pytest

from app.ingestion.chunker import chunk_pages


def test_chunker_preserves_source_metadata():
    pages = [
        {
            "text": "one two three four five six",
            "source": "sample.pdf",
            "page": 2,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert chunks[0]["text"] == "one two three four"
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["page"] == 2


def test_chunker_creates_overlapping_chunks():
    pages = [
        {
            "text": "one two three four five six",
            "source": "sample.pdf",
            "page": 1,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert len(chunks) == 2
    assert chunks[1]["text"] == "four five six"


def test_chunker_skips_empty_pages():
    pages = [
        {"text": "   ", "source": "empty.pdf", "page": 1},
        {"text": "hello world", "source": "sample.pdf", "page": 2},
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (4, -1),
        (4, 4),
        (4, 5),
    ],
)
def test_chunker_rejects_invalid_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [{"text": "some text", "source": "sample.pdf", "page": 1}],
            chunk_size=chunk_size,
            overlap=overlap,
        )
