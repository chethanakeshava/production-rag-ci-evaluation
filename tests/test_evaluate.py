import json

import pytest

from evaluation.evaluate import evaluate_answer, load_dataset, normalize_text


def test_evaluate_answer_passes_when_all_keywords_match():
    result = evaluate_answer(
        "The project uses FastAPI, React, PostgreSQL, and LangGraph.",
        ["FastAPI", "React", "PostgreSQL", "LangGraph"],
    )

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["missing_keywords"] == []


def test_evaluate_answer_fails_when_score_is_below_threshold():
    result = evaluate_answer(
        "The project uses React.",
        ["FastAPI", "React", "PostgreSQL", "LangGraph"],
    )

    assert result["score"] == 0.25
    assert result["passed"] is False
    assert result["missing_keywords"] == ["FastAPI", "PostgreSQL", "LangGraph"]


def test_evaluate_answer_is_case_insensitive():
    result = evaluate_answer(
        "The project uses fastapi.",
        ["FastAPI"],
    )

    assert result["score"] == 1.0
    assert result["passed"] is True


def test_evaluate_answer_handles_empty_keyword_list():
    result = evaluate_answer("Any answer", [])

    assert result["score"] == 1.0
    assert result["passed"] is True


def test_evaluate_answer_detects_model_name_in_answer():
    result = evaluate_answer(
        "The forecast model was XGBoost, with MAPE improving from 24.72% to 2.18%.",
        ["XGBoost", "24.72%", "2.18%"],
    )

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["missing_keywords"] == []


@pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014"])
def test_normalize_text_replaces_ascii_and_unicode_dashes(dash):
    assert normalize_text(f"state{dash}of{dash}the{dash}art") == ("state of the art")


def test_load_dataset_accepts_explicit_matching_path(tmp_path):
    dataset_path = tmp_path / "resume.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "resume-001",
                    "question": "What degree is listed?",
                    "expected_answer": "B.E.",
                    "expected_keywords": ["B.E."],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_dataset(dataset_path)[0]["id"] == "resume-001"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (None, "must be a JSON object"),
        ({"id": "x"}, "missing required fields"),
        (
            {
                "id": " ",
                "question": "Question?",
                "expected_answer": "Answer",
                "expected_keywords": ["answer"],
            },
            "'id' must be a non-empty string",
        ),
        (
            {
                "id": "x",
                "question": 3,
                "expected_answer": "Answer",
                "expected_keywords": ["answer"],
            },
            "'question' must be a non-empty string",
        ),
        (
            {
                "id": "x",
                "question": "Question?",
                "expected_answer": "",
                "expected_keywords": ["answer"],
            },
            "'expected_answer' must be a non-empty string",
        ),
        (
            {
                "id": "x",
                "question": "Question?",
                "expected_answer": "Answer",
                "expected_keywords": "answer",
            },
            "expected_keywords' must be a non-empty list",
        ),
        (
            {
                "id": "x",
                "question": "Question?",
                "expected_answer": "Answer",
                "expected_keywords": [],
            },
            "expected_keywords' must be a non-empty list",
        ),
        (
            {
                "id": "x",
                "question": "Question?",
                "expected_answer": "Answer",
                "expected_keywords": [" "],
            },
            "expected_keywords must contain non-empty strings",
        ),
    ],
)
def test_load_dataset_rejects_malformed_entries(tmp_path, entry, message):
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_dataset(dataset_path)


@pytest.mark.parametrize("dataset", [[], {}, "not a list"])
def test_load_dataset_rejects_invalid_top_level_values(tmp_path, dataset):
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_dataset(dataset_path)
