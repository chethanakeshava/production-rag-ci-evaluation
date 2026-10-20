
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": ""},
        )

    assert response.status_code == 422


def test_ask_returns_fallback_without_documents():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "What is in the documents?"},
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "don't have enough evidence" in response.json()["answer"]


def test_upload_rejects_non_pdf():
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
