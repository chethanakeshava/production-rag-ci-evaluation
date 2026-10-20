
from app.retrieval.embeddings import EmbeddingModel


def test_embedding_model_generates_vectors():
    model = EmbeddingModel()

    texts = [
        "Python is used for machine learning.",
        "Machine learning uses data to make predictions.",
    ]

    embeddings = model.encode(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_embedding_model_handles_empty_input():
    model = EmbeddingModel()

    assert model.encode([]) == []
