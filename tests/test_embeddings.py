import numpy as np

from app.retrieval.embeddings import EmbeddingModel


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.encoded_texts = None

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        self.encoded_texts = texts
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        return np.zeros((len(texts), 384), dtype=np.float32)


def test_embedding_model_generates_vectors(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.embeddings.SentenceTransformer",
        FakeSentenceTransformer,
    )
    model = EmbeddingModel()

    texts = [
        "Python is used for machine learning.",
        "Machine learning uses data to make predictions.",
    ]

    embeddings = model.encode(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert model.model.encoded_texts == texts


def test_embedding_model_handles_empty_input(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.embeddings.SentenceTransformer",
        FakeSentenceTransformer,
    )
    model = EmbeddingModel()

    assert model.encode([]) == []
