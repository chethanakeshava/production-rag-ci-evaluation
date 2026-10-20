import argparse
import json
import sys
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from evaluation.evaluate import evaluate_answer, load_dataset, normalize_text

MIN_CONTEXT_RECALL = 0.75


def evaluate_example(answer_evaluation: dict, context_recall: float) -> dict:
    """Pass only when answer quality and context recall meet their thresholds."""
    answer_passed = answer_evaluation["passed"]
    context_passed = context_recall >= MIN_CONTEXT_RECALL
    return {
        "answer_passed": answer_passed,
        "context_passed": context_passed,
        "passed": answer_passed and context_passed,
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
        context_evaluation = evaluate_context(
            retrieved_chunks, item["expected_keywords"]
        )
        context_matches = context_evaluation["matched"]
        context_recall = context_evaluation["recall"]
        print(f"Context recall: {context_recall:.2f}")
        print(f"Keywords found in context: {context_matches}")
        gate = evaluate_example(evaluation, context_recall)
        answer_passed = gate["answer_passed"]
        context_passed = gate["context_passed"]
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

    report_path = Path("evaluation/results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved to: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
