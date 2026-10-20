

from evaluation.evaluate import evaluate_answer


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
