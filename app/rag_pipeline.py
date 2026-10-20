
from pathlib import Path

from app.generation.answer_generator import AnswerGenerator
from app.ingestion.chunker import chunk_pages
from app.ingestion.document_loader import load_pdf
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class RAGPipeline:
    """Coordinate document ingestion, retrieval, reranking and generation."""
    def __init__(self) -> None:
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.documents_loaded = False
    def ingest_pdf(self, file_path: str | Path) -> int:
        """Load a PDF, chunk its pages and index the resulting chunks."""
        pages = load_pdf(file_path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("The PDF did not produce any text chunks")
        self.retriever.add_chunks(chunks)
        self.documents_loaded = True
        return len(chunks)
    def ask(
        self,
        question: str,
        top_k: int = 3,
        candidate_k: int = 10,
    ) -> dict:
        """Retrieve evidence, rerank it and generate an answer."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive")
        if not self.documents_loaded:
            return {
                "answer": (
                    "I don't have enough evidence in the provided "
                    "documents to answer this question."
                ),
                "sources": [],
            }
        candidates = self.retriever.search(
            question,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
        ranked_chunks = self.reranker.rerank(
            question,
            candidates,
            top_k=top_k,
        )
        answer = self.generator.generate(question, ranked_chunks)
        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "rerank_score": chunk["rerank_score"],
                "text": chunk["text"],
            }
            for chunk in ranked_chunks
        ]
        return {
            "answer": answer,
            "sources": sources,
        }
