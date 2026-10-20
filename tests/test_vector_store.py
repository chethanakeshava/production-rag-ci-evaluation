
import pytest

from app.retrieval.vector_store import VectorStore


class FakeEmbeddingModel:
    """Provide predictable embeddings without downloading a model."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "machine learning uses data": [1.0, 0.0],
            "neural networks learn patterns": [0.9, 0.1],
            "cooking pasta with sauce": [0.0, 1.0],
            "how does machine learning work": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


@pytest.fixture
def chunks():
    return [
        {
            "text": "machine learning uses data",
            "source": "ai.pdf",
            "page": 1,
        },
        {
            "text": "neural networks learn patterns",
            "source": "ai.pdf",
            "page": 2,
        },
        {
            "text": "cooking pasta with sauce",
            "source": "cooking.pdf",
            "page": 1,
        },
    ]


def test_search_returns_most_similar_chunk(chunks):
    store = VectorStore(FakeEmbeddingModel())
    store.add_chunks(chunks)

    results = store.search("how does machine learning work", top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "machine learning uses data"
    assert results[0]["source"] == "ai.pdf"
    assert results[0]["page"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_limits_results_to_available_chunks(chunks):
    store = VectorStore(FakeEmbeddingModel())
    store.add_chunks(chunks)

    assert len(store.search("how does machine learning work", top_k=10)) == 3


def test_search_returns_empty_when_index_is_empty():
    store = VectorStore(FakeEmbeddingModel())

    assert store.search("how does machine learning work") == []


def test_search_returns_empty_for_blank_query(chunks):
    store = VectorStore(FakeEmbeddingModel())
    store.add_chunks(chunks)

    assert store.search("   ") == []


def test_search_rejects_invalid_top_k():
    store = VectorStore(FakeEmbeddingModel())

    with pytest.raises(ValueError):
        store.search("machine learning", top_k=0)


def test_add_chunks_ignores_empty_input():
    store = VectorStore(FakeEmbeddingModel())
    store.add_chunks([])

    assert store.index is None
    assert store.chunks == []
