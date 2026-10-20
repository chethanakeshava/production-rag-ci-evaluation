from evaluation.evaluate import load_dataset
from evaluation.run_rag_evaluation import evaluate_context, evaluate_example


def test_quality_gate_passes_when_answer_and_context_pass():
    answer_evaluation = {"passed": True, "score": 1.0}

    result = evaluate_example(answer_evaluation, 1.0)

    assert result["answer_passed"] is True
    assert result["context_passed"] is True
    assert result["passed"] is True


def test_quality_gate_fails_when_context_recall_is_below_threshold():
    answer_evaluation = {"passed": True, "score": 1.0}

    result = evaluate_example(answer_evaluation, 0.50)

    assert result["answer_passed"] is True
    assert result["context_passed"] is False
    assert result["passed"] is False


def test_quality_gate_fails_when_answer_fails():
    answer_evaluation = {"passed": False, "score": 0.50}

    result = evaluate_example(answer_evaluation, 1.0)

    assert result["answer_passed"] is False
    assert result["context_passed"] is True
    assert result["passed"] is False


def test_quality_gate_accepts_context_at_threshold():
    answer_evaluation = {"passed": True, "score": 1.0}

    result = evaluate_example(answer_evaluation, 0.75)

    assert result["context_passed"] is True
    assert result["passed"] is True


def test_context_evaluation_matches_case_insensitively():
    result = evaluate_context(
        [{"text": "FastAPI serves this application."}], ["FastAPI", "FAISS"]
    )

    assert result == {
        "recall": 0.5,
        "matched": ["FastAPI"],
        "missing": ["FAISS"],
    }


def test_context_evaluation_uses_retrieved_text_not_dataset_context():
    result = evaluate_context(
        [{"text": "The resume mentions FastAPI."}],
        ["FastAPI", "LangChain", "FAISS", "Google Gemini"],
    )

    assert result["recall"] == 0.25
    assert result["matched"] == ["FastAPI"]
    assert result["missing"] == ["LangChain", "FAISS", "Google Gemini"]


def test_resume_dataset_questions_and_keywords_are_document_grounded():
    dataset = load_dataset()

    all_expected_text = " ".join(
        item["question"]
        + " "
        + item["expected_answer"]
        + " "
        + " ".join(item["expected_keywords"])
        for item in dataset
    )

    assert "RAG Chatbot" not in all_expected_text
    assert "Google Gemini" not in all_expected_text
    assert all("context" not in item for item in dataset)
    assert all(
        {"id", "question", "expected_answer", "expected_keywords"} == set(item)
        for item in dataset
    )
