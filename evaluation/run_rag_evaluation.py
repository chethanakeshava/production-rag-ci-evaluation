import argparse
import json
import re
import sys
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from evaluation.evaluate import evaluate_answer, load_dataset, normalize_text
from evaluation.groundedness import evaluate_groundedness

MIN_CONTEXT_RECALL = 0.75
RESULTS_PATH = Path("evaluation/results.json")


def evaluate_example(
    answer_evaluation: dict,
    context_recall: float,
    groundedness_evaluation: dict | None = None,
) -> dict:
    """Pass only when answer, context, and groundedness checks pass."""
    answer_passed = answer_evaluation["passed"]
    context_passed = context_recall >= MIN_CONTEXT_RECALL

    groundedness_passed = (
        groundedness_evaluation["passed"]
        if groundedness_evaluation is not None
        else True
    )

    return {
        "answer_passed": answer_passed,
        "context_passed": context_passed,
        "groundedness_passed": groundedness_passed,
        "passed": (answer_passed and context_passed and groundedness_passed),
    }


def evaluate_context(chunks: list[dict], expected_keywords: list[str]) -> dict:
    """Measure expected keyword coverage in the actual retrieved evidence."""
    retrieved_text = normalize_text(" ".join(chunk.get("text", "") for chunk in chunks))
    matched = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in retrieved_text
    ]
    missing = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) not in retrieved_text
    ]
    recall = len(matched) / len(expected_keywords) if expected_keywords else 1.0
    return {"recall": recall, "matched": matched, "missing": missing}


def split_answer_claims(answer: str) -> list[str]:
    """Use answer sentences as deterministic candidate groundedness claims."""
    without_citations = re.sub(r"\[\d+\]", "", answer)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", without_citations)
        if sentence.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers against a document-matched dataset."
    )
    parser.add_argument("pdf", type=Path, help="PDF document to index")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Evaluation dataset JSON matched to the supplied PDF",
    )
    args = parser.parse_args()
    pdf_path = args.pdf
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        print(f"Error: PDF file not found or invalid: {pdf_path}")
        return 2

    try:
        dataset = load_dataset(args.dataset)
    except (OSError, TypeError, ValueError) as error:
        print(f"Error loading evaluation dataset {args.dataset}: {error}")
        return 2
    # Initialize the pipeline and index the PDF once.
    pipeline = RAGPipeline()
    chunk_count = pipeline.ingest_pdf(pdf_path)
    print(f"Indexed {chunk_count} chunks from {pdf_path.name}")
    print(f"Using dataset {args.dataset}")
    print(f"Evaluating {len(dataset)} questions...\n")
    report = []
    failures = 0
    for item in dataset:
        result = pipeline.ask(item["question"])
        answer = result["answer"]
        retrieved_chunks = result.get("sources", [])
        evaluation = evaluate_answer(answer, item["expected_keywords"])
        answer_claims = split_answer_claims(answer)
        context_evaluation = evaluate_context(
            retrieved_chunks, item["expected_keywords"]
        )
        groundedness_evaluation = evaluate_groundedness(
            answer_claims,
            retrieved_chunks,
        )
        context_matches = context_evaluation["matched"]
        context_recall = context_evaluation["recall"]
        print(f"Context recall: {context_recall:.2f}")
        print(f"Keywords found in context: {context_matches}")
        print(f"Groundedness: {groundedness_evaluation['score']:.2f}")
        print(
            "Unsupported answer claims: "
            f"{groundedness_evaluation['unsupported_claims']}"
        )
        gate = evaluate_example(
            evaluation,
            context_recall,
            groundedness_evaluation,
        )
        answer_passed = gate["answer_passed"]
        context_passed = gate["context_passed"]
        groundedness_passed = gate["groundedness_passed"]
        example_passed = gate["passed"]
        status = "PASS" if example_passed else "FAIL"
        print(f"{item['id']}: {status}")
        print(f"Score: {evaluation['score']:.2f}")
        print(f"Answer: {answer}")
        print(f"Missing keywords: {evaluation['missing_keywords']}\n")
        report.append(
            {
                "id": item["id"],
                "dataset": str(args.dataset),
                "answer_score": evaluation["score"],
                "answer_passed": answer_passed,
                "context_recall": context_recall,
                "context_matches": context_matches,
                "context_missing_keywords": context_evaluation["missing"],
                "missing_keywords": evaluation["missing_keywords"],
                "groundedness_score": groundedness_evaluation["score"],
                "groundedness_passed": groundedness_passed,
                "supported_claims": groundedness_evaluation["supported_claims"],
                "unsupported_claims": groundedness_evaluation["unsupported_claims"],
                "retrieved_sources": [
                    {
                        "source": chunk.get("source"),
                        "page": chunk.get("page"),
                        "text": chunk.get("text", ""),
                    }
                    for chunk in retrieved_chunks
                ],
                "context_passed": context_passed,
                "passed": example_passed,
            }
        )
        if not example_passed:
            failures += 1
    print(
        f"Final results: {len(dataset) - failures} passed, "
        f"{failures} failed out of {len(dataset)}."
    )

    report_path = RESULTS_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved to: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
