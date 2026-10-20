
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def make_test_pipeline():
    pipeline = MagicMock()
    pipeline.ask.return_value = {
        "answer": (
            "I don't have enough evidence in the provided "
            "documents to answer this question."
        ),
        "sources": [],
    }
    pipeline.ingest_pdf.return_value = 2
    return pipeline


def test_health_endpoint():
    pipeline = make_test_pipeline()

    with (
        patch("app.main.RAGPipeline", return_value=pipeline),
        TestClient(app) as client,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    pipeline = make_test_pipeline()

    with (
        patch("app.main.RAGPipeline", return_value=pipeline),
        TestClient(app) as client,
    ):
        response = client.post(
            "/ask",
            json={"question": ""},
        )

    assert response.status_code == 422
    pipeline.ask.assert_not_called()


def test_ask_returns_fallback_without_documents():
    pipeline = make_test_pipeline()

    with (
        patch("app.main.RAGPipeline", return_value=pipeline),
        TestClient(app) as client,
    ):
        response = client.post(
            "/ask",
            json={"question": "What is in the documents?"},
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "don't have enough evidence" in response.json()["answer"]


def test_upload_rejects_non_pdf():
    pipeline = make_test_pipeline()

    with (
        patch("app.main.RAGPipeline", return_value=pipeline),
        TestClient(app) as client,
    ):
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
    pipeline.ingest_pdf.assert_not_called()
