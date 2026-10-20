
import pytest

from app.retrieval.reranker import Reranker


class FakeCrossEncoder:
    """Return predictable relevance scores for tests."""

    def predict(self, pairs):
        score_by_text = {
            "cooking pasta with sauce": 0.1,
            "neural networks learn patterns": 0.6,
            "machine learning uses data": 0.9,
        }

        return [
            score_by_text[text]
            for _, text in pairs
        ]


@pytest.fixture
def documents():
    return [
        {
            "text": "cooking pasta with sauce",
            "source": "cooking.pdf",
            "page": 1,
        },
        {
            "text": "neural networks learn patterns",
            "source": "ai.pdf",
            "page": 2,
        },
        {
            "text": "machine learning uses data",
            "source": "ai.pdf",
            "page": 1,
        },
    ]


@pytest.fixture
def reranker():
    return Reranker(model=FakeCrossEncoder())


def test_reranker_sorts_by_relevance(reranker, documents):
    results = reranker.rerank("machine learning", documents)

    assert results[0]["text"] == "machine learning uses data"
    assert results[1]["text"] == "neural networks learn patterns"
    assert results[2]["text"] == "cooking pasta with sauce"


def test_reranker_limits_results(reranker, documents):
    results = reranker.rerank(
        "machine learning",
        documents,
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_preserves_metadata(reranker, documents):
    results = reranker.rerank("machine learning", documents)

    assert results[0]["source"] == "ai.pdf"
    assert results[0]["page"] == 1
    assert results[0]["rerank_score"] == pytest.approx(0.9)


def test_reranker_returns_empty_for_blank_query(reranker, documents):
    assert reranker.rerank("   ", documents) == []


def test_reranker_returns_empty_for_no_documents(reranker):
    assert reranker.rerank("machine learning", []) == []


def test_reranker_rejects_invalid_top_k(reranker, documents):
    with pytest.raises(ValueError):
        reranker.rerank("machine learning", documents, top_k=0)
