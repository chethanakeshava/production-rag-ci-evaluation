import json
import sys
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from evaluation.evaluate import evaluate_answer, load_dataset


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m evaluation.run_rag_evaluation "
            '"path/to/document.pdf"'
        )
        return 2

    pdf_path = Path(sys.argv[1])

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        print(f"Error: PDF file not found or invalid: {pdf_path}")
        return 2

    dataset = load_dataset()

    # Initialize the pipeline and index the PDF once.
    pipeline = RAGPipeline()
    chunk_count = pipeline.ingest_pdf(pdf_path)

    print(f"Indexed {chunk_count} chunks from {pdf_path.name}")
    print(f"Evaluating {len(dataset)} questions...\n")
    report = []
    failures = 0

    for item in dataset:
        result = pipeline.ask(item["question"])
        answer = result["answer"]
        retrieved_chunks = result.get("sources", [])
        evaluation = evaluate_answer(answer, item["expected_keywords"])
        retrieved_text = " ".join(
            chunk.get("text", "") for chunk in retrieved_chunks
        ).casefold()
        context_matches = [
            keyword
            for keyword in item["expected_keywords"]
            if keyword.casefold() in retrieved_text
        ]
        context_recall = (
            len(context_matches) / len(item["expected_keywords"])
            if item["expected_keywords"]
            else 1.0
        )
        print(f"Context recall: {context_recall:.2f}")
        print(f"Keywords found in context: {context_matches}")
        status = "PASS" if evaluation["passed"] else "FAIL"
        print(f"{item['id']}: {status}")
        print(f"Score: {evaluation['score']:.2f}")
        print(f"Answer: {answer}")
        print(f"Missing keywords: {evaluation['missing_keywords']}\n")
        report.append({
            "id": item["id"],
            "answer_score": evaluation["score"],
            "answer_passed": evaluation["passed"],
            "context_recall": context_recall,
            "context_matches": context_matches,
            "missing_keywords": evaluation["missing_keywords"],
        })
        if not evaluation["passed"]:
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
