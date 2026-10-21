
import argparse
import json
import time
from pathlib import Path

from evaluation.benchmark.metrics import mean_reciprocal_rank, recall_at_k

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "corpus.json"
DATASET_PATH = BASE_DIR / "dataset.json"
TOP_K = 3
MIN_RECALL = 0.75


def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")

    return data


def validate_benchmark_data(
    corpus: list[dict],
    dataset: list[dict],
) -> None:
    """Validate corpus and relevance labels before running retrieval."""
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must contain at least one chunk")
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must contain at least one question")

    corpus_ids = []
    for index, chunk in enumerate(corpus):
        if not isinstance(chunk, dict):
            raise TypeError(f"Corpus chunk at index {index} must be an object")
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Corpus chunk at index {index} must have a non-empty string id"
            )
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(f"Corpus chunk {chunk_id!r} must have non-empty text")
        corpus_ids.append(chunk_id)

    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Corpus chunk IDs must be unique")

    question_ids = []
    corpus_id_set = set(corpus_ids)
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise TypeError(f"Dataset question at index {index} must be an object")
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(
                f"Dataset question at index {index} must have a non-empty string id"
            )
        question_ids.append(question_id)

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {question_id!r} must be non-empty")

        relevant_ids = item.get("relevant_chunk_ids")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(chunk_id, str) for chunk_id in relevant_ids)
        ):
            raise ValueError(
                f"Question {question_id!r} must have a non-empty list of relevant_chunk_ids"
            )
        unknown_ids = [
            chunk_id for chunk_id in relevant_ids if chunk_id not in corpus_id_set
        ]
        if unknown_ids:
            raise ValueError(
                f"Question {question_id!r} references unknown corpus chunk IDs: "
                f"{unknown_ids}"
            )

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Dataset question IDs must be unique")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval quality")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=BASE_DIR / "results.json")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Run deterministic BM25 evaluation without downloading ML models",
    )
    args = parser.parse_args()
    return run_benchmark(
        args.corpus,
        args.dataset,
        args.report,
        use_models=not args.skip_models,
    )


def run_benchmark(
    corpus_path: Path = CORPUS_PATH,
    dataset_path: Path = DATASET_PATH,
    report_path: Path = BASE_DIR / "results.json",
    *,
    use_models: bool = True,
    model_components: dict | None = None,
) -> int:
    """Run retrieval metrics against a labeled corpus and enforce recall gate."""
    corpus = load_json(corpus_path)
    dataset = load_json(dataset_path)
    validate_benchmark_data(corpus, dataset)

    from rank_bm25 import BM25Okapi

    if use_models and model_components is None:
        from app.retrieval.embeddings import EmbeddingModel
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.retrieval.reranker import Reranker
        from app.retrieval.vector_store import VectorStore

        embedding_model = EmbeddingModel()
        vector_store = VectorStore(embedding_model)
        vector_store.add_chunks(corpus)
        reranker = Reranker()
        embedding_model.encode(["benchmark warmup"])
        vector_store.search(dataset[0]["question"], top_k=TOP_K)
        model_components = {
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "reranker": reranker,
        }

    if use_models:
        from app.retrieval.hybrid_retriever import HybridRetriever

        embedding_model = model_components["embedding_model"]
        vector_store = model_components["vector_store"]
        reranker = model_components["reranker"]
        hybrid_factory = model_components.get(
            "hybrid_retriever_factory",
            HybridRetriever,
        )
        hybrid = hybrid_factory(vector_store)
        hybrid.chunks = corpus.copy()
        hybrid.bm25 = BM25Okapi([hybrid._tokenize(chunk["text"]) for chunk in corpus])
        tokenize = hybrid._tokenize
        bm25 = hybrid.bm25
    else:
        hybrid = None
        def tokenize(text: str) -> list[str]:
            return text.lower().split()

        bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in corpus])

    strategies = {"bm25": []}
    latencies = {"bm25": []}
    if use_models:
        strategies.update({"semantic": [], "hybrid": [], "hybrid_reranked": []})
        latencies.update({"semantic": [], "hybrid": [], "hybrid_reranked": []})
    relevant_lists = []

    for item in dataset:
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        relevant_lists.append(relevant_ids)

        start = time.perf_counter()
        tokens = tokenize(question)
        scores = bm25.get_scores(tokens)
        bm25_indices = sorted(
            range(len(corpus)),
            key=lambda index: scores[index],
            reverse=True,
        )[:TOP_K]
        bm25_results = [corpus[index] for index in bm25_indices]
        latencies["bm25"].append(time.perf_counter() - start)

        results_by_strategy = {"bm25": bm25_results}
        if use_models:
            start = time.perf_counter()
            semantic_results = vector_store.search(question, top_k=TOP_K)
            latencies["semantic"].append(time.perf_counter() - start)
            start = time.perf_counter()
            hybrid_results = hybrid.search(question, top_k=TOP_K, candidate_k=len(corpus))
            latencies["hybrid"].append(time.perf_counter() - start)
            start = time.perf_counter()
            candidates = hybrid.search(question, top_k=len(corpus), candidate_k=len(corpus))
            reranked_results = reranker.rerank(question, candidates, top_k=TOP_K)
            latencies["hybrid_reranked"].append(time.perf_counter() - start)
            results_by_strategy.update({
                "semantic": semantic_results,
                "hybrid": hybrid_results,
                "hybrid_reranked": reranked_results,
            })

        for name, results in results_by_strategy.items():
            strategies[name].append(
                [result["id"] for result in results]
            )

    print(f"Benchmark questions: {len(dataset)}")
    print(f"Corpus chunks: {len(corpus)}")
    print(f"Metrics: Recall@{TOP_K}, MRR, average latency (ms)\n")

    report = {}
    for name, ranked_lists in strategies.items():
        recall_scores = [
            recall_at_k(ranked, relevant, TOP_K)
            for ranked, relevant in zip(ranked_lists, relevant_lists)
        ]
        mrr = mean_reciprocal_rank(ranked_lists, relevant_lists)
        average_latency_ms = (
            sum(latencies[name]) / len(latencies[name]) * 1000
        )

        report[name] = {
            f"recall_at_{TOP_K}": sum(recall_scores) / len(recall_scores),
            "mrr": mrr,
            "average_latency_ms": average_latency_ms,
        }

        print(
            f"{name:20} "
            f"Recall@{TOP_K}={report[name][f'recall_at_{TOP_K}']:.3f} "
            f"MRR={mrr:.3f} "
            f"latency={average_latency_ms:.2f} ms"
        )

    gate_passed = report["bm25"][f"recall_at_{TOP_K}"] >= MIN_RECALL
    report["gate"] = {"minimum_recall": MIN_RECALL, "passed": gate_passed}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved benchmark report to {report_path}")
    print(f"Retrieval quality gate: {'PASS' if gate_passed else 'FAIL'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
