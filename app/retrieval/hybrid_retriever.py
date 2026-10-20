
from rank_bm25 import BM25Okapi

from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Combine semantic search and keyword search using RRF."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.chunks: list[dict] = []
        self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase word tokens."""
        return text.lower().split()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunks to both semantic and keyword search indexes."""
        if not chunks:
            return

        self.vector_store.add_chunks(chunks)
        self.chunks.extend(chunks)

        corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Fuse FAISS and BM25 rankings using Reciprocal Rank Fusion."""
        if top_k <= 0 or candidate_k <= 0 or rrf_k <= 0:
            raise ValueError("top_k, candidate_k and rrf_k must be positive")

        if not query.strip() or not self.chunks or self.bm25 is None:
            return []

        candidate_count = min(candidate_k, len(self.chunks))

        semantic_results = self.vector_store.search(
            query,
            top_k=candidate_count,
        )

        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        keyword_indices = sorted(
            range(len(self.chunks)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_count]

        fused = {}

        # Rank positions start at 1 in Reciprocal Rank Fusion.
        for rank, result in enumerate(semantic_results, start=1):
            key = (result["source"], result["page"], result["text"])
            fused.setdefault(key, result.copy())
            fused[key]["rrf_score"] = (
                fused[key].get("rrf_score", 0.0)
                + 1.0 / (rrf_k + rank)
            )

        for rank, chunk_index in enumerate(keyword_indices, start=1):
            chunk = self.chunks[chunk_index]
            key = (chunk["source"], chunk["page"], chunk["text"])

            fused.setdefault(key, chunk.copy())
            fused[key]["rrf_score"] = (
                fused[key].get("rrf_score", 0.0)
                + 1.0 / (rrf_k + rank)
            )

        results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return results[:top_k]
