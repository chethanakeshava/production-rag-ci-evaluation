
import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore


class FakeEmbeddingModel:
    """Return predictable vectors for testing."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "machine learning uses data": [1.0, 0.0],
            "neural networks learn patterns": [0.9, 0.1],
            "cooking pasta with sauce": [0.0, 1.0],
            "machine learning": [1.0, 0.0],
            "unknown query": [0.0, 1.0],
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
@pytest.fixture
def retriever(chunks):
    vector_store = VectorStore(FakeEmbeddingModel())
    hybrid = HybridRetriever(vector_store)
    hybrid.add_chunks(chunks)
    return hybrid
def test_hybrid_search_returns_relevant_chunks(retriever):
    results = retriever.search("machine learning", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "machine learning uses data"
    assert results[0]["rrf_score"] > 0
def test_hybrid_search_preserves_source_metadata(retriever):
    results = retriever.search("machine learning", top_k=3)
    first_result = results[0]
    assert first_result["source"] == "ai.pdf"
    assert first_result["page"] == 1
def test_hybrid_search_returns_empty_for_blank_query(retriever):
    assert retriever.search("   ") == []
def test_hybrid_search_returns_empty_without_chunks():
    retriever = HybridRetriever(VectorStore(FakeEmbeddingModel()))
    assert retriever.search("machine learning") == []
@pytest.mark.parametrize(
    ("top_k", "candidate_k", "rrf_k"),
    [
        (0, 10, 60),
        (3, 0, 60),
        (3, 10, 0),
    ],
)
def test_hybrid_search_rejects_invalid_parameters(
    retriever,
    top_k,
    candidate_k,
    rrf_k,
):
    with pytest.raises(ValueError):
        retriever.search(
            "machine learning",
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )