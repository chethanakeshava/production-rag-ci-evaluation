import json
import re
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.json"
MIN_KEYWORD_SCORE = 0.75


def normalize_text(text: str) -> str:
    """Normalize case, punctuation, hyphens, and whitespace."""
    text = text.casefold()
    text = re.sub(r"[-\u2013\u2014]", " ", text)
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


def load_dataset(dataset_path: str | Path = DATASET_PATH) -> list[dict]:
    """Load and validate the evaluation dataset."""
    with Path(dataset_path).open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    required_fields = {"id", "question", "expected_answer", "expected_keywords"}
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise TypeError(f"Dataset example {index} must be a JSON object.")

        missing_fields = required_fields - item.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Dataset example {index} is missing required fields: {missing}."
            )

        for field in ("id", "question", "expected_answer"):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Dataset example {index} field '{field}' must be a "
                    "non-empty string."
                )

        keywords = item["expected_keywords"]
        if not isinstance(keywords, list) or not keywords:
            raise ValueError(
                f"Dataset example {index} field 'expected_keywords' "
                "must be a non-empty list."
            )
        if any(
            not isinstance(keyword, str) or not keyword.strip() for keyword in keywords
        ):
            raise ValueError(
                f"Dataset example {index} expected_keywords must contain "
                "non-empty strings."
            )

    return dataset


def main() -> int:
    dataset = load_dataset()
    failures = 0

    print(f"Loaded {len(dataset)} evaluation examples.")

    for item in dataset:
        # Score the dataset's reference answer independently from RAG retrieval.
        answer = item["expected_answer"]
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
