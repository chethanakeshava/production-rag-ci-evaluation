
from unittest.mock import MagicMock

import pytest

from app.generation.answer_generator import AnswerGenerator


def test_generator_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.generation.answer_generator.load_dotenv",
        lambda: None,
    )

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        AnswerGenerator()


def test_generator_rejects_empty_question(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.generation.answer_generator.load_dotenv",
        lambda: None,
    )

    generator = AnswerGenerator()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate("   ", [])


def test_generator_returns_fallback_without_chunks(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.generation.answer_generator.load_dotenv",
        lambda: None,
    )

    generator = AnswerGenerator()

    answer = generator.generate("What does the document say?", [])

    assert answer == (
        "I don't have enough evidence in the provided documents "
        "to answer this question."
    )


def test_generator_uses_document_context(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.generation.answer_generator.load_dotenv",
        lambda: None,
    )

    generator = AnswerGenerator()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "The resume mentions DharaaAI and Supply Chain Demand Forecasting. [1]"
    )

    mock_create = MagicMock(return_value=mock_response)
    generator.client.chat.completions.create = mock_create

    chunks = [
        {
            "text": "Projects: DharaaAI and Supply Chain Demand Forecasting.",
            "source": "resume.pdf",
            "page": 1,
        }
    ]

    answer = generator.generate("What projects are mentioned?", chunks)

    assert "DharaaAI" in answer
    assert "Supply Chain Demand Forecasting" in answer

    call_args = mock_create.call_args.kwargs
    prompt_content = call_args["messages"][1]["content"]

    assert "DharaaAI" in prompt_content
    assert "Supply Chain Demand Forecasting" in prompt_content
    assert "What projects are mentioned?" in prompt_content
def test_generator_retries_when_answer_omits_relevant_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.generation.answer_generator.load_dotenv",
        lambda: None,
    )

    generator = AnswerGenerator()

    first_response = MagicMock()
    first_response.choices[0].message.content = (
        "MAPE decreased from 24.72% to 2.18%."
    )

    corrected_response = MagicMock()
    corrected_response.choices[0].message.content = (
        "The project used XGBoost and reduced MAPE from 24.72% to 2.18%. [1]"
    )

    generator.client.chat.completions.create = MagicMock(
        side_effect=[first_response, corrected_response]
    )

    chunks = [
        {
            "text": (
                "The project used XGBoost and reduced MAPE "
                "from 24.72% to 2.18%."
            ),
            "source": "resume.pdf",
            "page": 1,
        }
    ]

    answer = generator.generate(
        "What improvement did the Supply Chain Demand Forecasting project achieve?",
        chunks,
    )

    assert "XGBoost" in answer
    assert "24.72%" in answer
    assert "2.18%" in answer
    assert generator.client.chat.completions.create.call_count == 2