import json

from evaluation.evaluate import load_dataset
from evaluation.run_rag_evaluation import (
    evaluate_context,
    evaluate_example,
    main,
    split_answer_claims,
)


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


def test_quality_gate_fails_when_groundedness_fails():
    answer_evaluation = {"passed": True, "score": 1.0}
    groundedness_evaluation = {"passed": False, "score": 0.0}

    result = evaluate_example(
        answer_evaluation,
        1.0,
        groundedness_evaluation,
    )

    assert result["answer_passed"] is True
    assert result["context_passed"] is True
    assert result["groundedness_passed"] is False
    assert result["passed"] is False


def test_runner_scores_generated_answer_against_retrieved_chunks(monkeypatch, tmp_path):
    import evaluation.run_rag_evaluation as runner

    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"placeholder PDF; pipeline is mocked")
    result_path = tmp_path / "results.json"
    answer = "FastAPI is used. Django powers another service."
    retrieved_chunk = {
        "source": "document.pdf",
        "page": 1,
        "text": "FastAPI is used.",
    }

    class FakePipeline:
        def ingest_pdf(self, path):
            return 1

        def ask(self, question):
            return {"answer": answer, "sources": [retrieved_chunk]}

    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda path: [
            {
                "id": "integration-001",
                "question": "What framework?",
                "expected_answer": "FastAPI",
                "expected_keywords": ["FastAPI"],
            }
        ],
    )
    monkeypatch.setattr(runner, "RAGPipeline", FakePipeline)
    monkeypatch.setattr(runner, "RESULTS_PATH", result_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_rag_evaluation", str(pdf_path), "--dataset", "resume.json"],
    )

    exit_code = main()

    assert exit_code == 1
    report = json.loads(result_path.read_text(encoding="utf-8"))
    example = report[0]
    assert example["context_recall"] == 1.0
    assert example["groundedness_score"] == 0.5
    assert example["groundedness_passed"] is False
    assert example["supported_claims"] == ["FastAPI is used."]
    assert example["unsupported_claims"] == ["Django powers another service."]
    assert example["retrieved_sources"][0]["text"] == retrieved_chunk["text"]
    assert example["passed"] is False


def test_split_answer_claims_handles_sentences_citations_and_decimals():
    answer = "MAPE fell from 24.72% to 2.18%. [1] XGBoost was used."

    assert split_answer_claims(answer) == [
        "MAPE fell from 24.72% to 2.18%.",
        "XGBoost was used.",
    ]


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
