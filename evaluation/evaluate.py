
import json
import re
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.json"
MIN_KEYWORD_SCORE = 0.75
def normalize_text(text: str) -> str:
    """Normalize case, punctuation, hyphens, and whitespace."""
    text = text.casefold()
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def evaluate_answer(
    answer: str,
    expected_keywords: list[str],
) -> dict:
    """Score an answer by the fraction of expected keywords it contains."""
    normalized_answer = normalize_text(answer)

    matched = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_answer
    ]

    missing = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) not in normalized_answer
    ]

    score = len(matched) / len(expected_keywords) if expected_keywords else 1.0

    return {
        "score": score,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "passed": score >= MIN_KEYWORD_SCORE,
    }


def load_dataset() -> list[dict]:
    """Load and validate the evaluation dataset."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for item in dataset:
        for field in ("id", "question", "expected_keywords"):
            if field not in item:
                raise ValueError(f"Dataset example is missing field: {field}")

    return dataset


def main() -> int:
    dataset = load_dataset()
    failures = 0

    print(f"Loaded {len(dataset)} evaluation examples.")

    for item in dataset:
        # Temporary expected-answer baseline until actual RAG inference
        # is connected in a later step.
        answer = item.get("expected_answer", "")
        result = evaluate_answer(answer, item["expected_keywords"])

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{item['id']}: {status} | "
            f"score={result['score']:.2f} | "
            f"missing={result['missing_keywords']}"
        )

        if not result["passed"]:
            failures += 1

    print(f"\nResults: {len(dataset) - failures} passed, {failures} failed.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
