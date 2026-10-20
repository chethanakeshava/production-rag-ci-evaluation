
from sentence_transformers import CrossEncoder


class Reranker:
    """Rerank retrieved document chunks using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
    ) -> None:
        self.model = model or CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Return document chunks ordered by cross-encoder relevance."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query.strip() or not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_documents = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            ranked_documents.append(result)

        ranked_documents.sort(
            key=lambda document: document["rerank_score"],
            reverse=True,
        )

        return ranked_documents[:top_k]
