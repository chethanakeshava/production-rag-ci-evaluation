from unittest.mock import patch

import pymupdf

from app.rag_pipeline import RAGPipeline


class FakeReranker:
    def rerank(self, query, documents, top_k=3):
        results = []
        for document in documents[:top_k]:
            result = document.copy()
            result["rerank_score"] = 0.9
            results.append(result)
        return results


class FakeGenerator:
    def generate(self, question, chunks):
        if not chunks:
            return (
                "I don't have enough evidence in the provided "
                "documents to answer this question."
            )
        return f"Test answer based on {chunks[0]['source']}."


def create_test_pdf(path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "The RAG pipeline retrieves relevant document evidence. "
            "FAISS supports vector similarity search.",
        )
        document.save(path)


def test_pipeline_ingests_pdf_and_returns_sources(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    with patch("app.rag_pipeline.AnswerGenerator"):
         pipeline = RAGPipeline()

    pipeline.reranker = FakeReranker()
    pipeline.generator = FakeGenerator()

    chunk_count = pipeline.ingest_pdf(pdf_path)
    result = pipeline.ask("What does the pipeline retrieve?")

    assert chunk_count > 0
    assert "Test answer" in result["answer"]
    assert len(result["sources"]) > 0
    assert result["sources"][0]["source"] == "sample.pdf"
    assert result["sources"][0]["page"] == 1


def test_pipeline_returns_fallback_without_documents():
   with patch("app.rag_pipeline.AnswerGenerator"):
    pipeline = RAGPipeline()

    pipeline.reranker = FakeReranker()
    pipeline.generator = FakeGenerator()

    result = pipeline.ask("What is in the documents?")

    assert "don't have enough evidence" in result["answer"]
    assert result["sources"] == []

def test_pipeline_rejects_empty_question(monkeypatch):
    import pytest

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    with patch("app.rag_pipeline.AnswerGenerator"):
         pipeline = RAGPipeline()

    pipeline.reranker = FakeReranker()
    pipeline.generator = FakeGenerator()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("   ")
