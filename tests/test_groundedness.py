from evaluation.groundedness import evaluate_groundedness


def test_groundedness_passes_when_claim_matches_evidence():
    result = evaluate_groundedness(
        ["FastAPI is the backend framework"],
        [{"text": "FastAPI is the backend framework"}],
    )

    assert result["score"] == 1.0
    assert result["supported_claims"] == ["FastAPI is the backend framework"]
    assert result["unsupported_claims"] == []
    assert result["passed"] is True


def test_groundedness_accepts_normalized_hyphenation_variant():
    result = evaluate_groundedness(
        ["FastAPI is used for a state of the art service."],
        [{"text": "FastAPI is used for a state-of-the-art service."}],
    )

    assert result["supported_claims"] == [
        "FastAPI is used for a state of the art service."
    ]
    assert result["unsupported_claims"] == []
    assert result["passed"] is True


def test_groundedness_fails_when_claim_is_not_in_evidence():
    result = evaluate_groundedness(
        ["Django is the backend framework"],
        [{"text": "FastAPI is the backend framework"}],
    )

    assert result["score"] == 0.0
    assert result["unsupported_claims"] == ["Django is the backend framework"]
    assert result["passed"] is False


def test_groundedness_does_not_assume_paraphrase_is_semantically_supported():
    result = evaluate_groundedness(
        ["FastAPI powers the backend"],
        [{"text": "FastAPI is used as the web framework"}],
    )

    assert result["supported_claims"] == []
    assert result["unsupported_claims"] == ["FastAPI powers the backend"]
    assert result["passed"] is False


def test_groundedness_marks_direct_contradiction_as_unsupported():
    result = evaluate_groundedness(
        ["Django is not used; FastAPI is not the backend framework"],
        [{"text": "FastAPI is the backend framework; Django is not used."}],
    )

    assert result["supported_claims"] == []
    assert len(result["unsupported_claims"]) == 1
    assert result["passed"] is False


def test_groundedness_scores_multiple_claims():
    result = evaluate_groundedness(
        [
            "FastAPI is the backend framework",
            "PostgreSQL stores the data",
        ],
        [{"text": "FastAPI is the backend framework"}],
    )

    assert result["score"] == 0.5
    assert len(result["supported_claims"]) == 1
    assert len(result["unsupported_claims"]) == 1
    assert result["passed"] is False


def test_groundedness_fails_when_evidence_is_empty():
    result = evaluate_groundedness(
        ["FastAPI is the backend framework"],
        [],
    )

    assert result["score"] == 0.0
    assert result["passed"] is False


def test_groundedness_accepts_exact_threshold():
    result = evaluate_groundedness(
        ["FastAPI", "PostgreSQL", "FAISS", "Django"],
        [{"text": ("FastAPI PostgreSQL FAISS")}],
    )

    assert result["score"] == 0.75
    assert result["passed"] is True


def test_groundedness_handles_no_claims():
    result = evaluate_groundedness([], [])

    assert result["score"] == 1.0
    assert result["passed"] is True
