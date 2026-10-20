
import faiss
import numpy as np

from app.retrieval.embeddings import EmbeddingModel


class VectorStore:
    """Store and retrieve document chunks using FAISS."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index = None
        self.chunks: list[dict] = []

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed and add document chunks to the vector index."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a 2D array with nonzero dimensions")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values")

        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        elif vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the index")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the most similar chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query.strip() or self.index is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        query_vector = np.asarray(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = self.chunks[int(index)].copy()
            result["score"] = float(score)
            results.append(result)

        return results
