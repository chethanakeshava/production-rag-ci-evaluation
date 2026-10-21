
import json

import pytest

from evaluation.benchmark.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.benchmark.run_benchmark import (
    run_benchmark,
    validate_benchmark_data,
)


def test_recall_at_k_finds_relevant_chunk():
    assert recall_at_k(
        ["wrong", "api-framework", "other"],
        ["api-framework"],
        2,
    ) == 1.0


def test_recall_at_k_is_zero_when_no_relevant_chunk_is_found():
    assert recall_at_k(["wrong", "other"], ["api-framework"], 2) == 0.0


def test_recall_at_k_handles_multiple_relevant_chunks():
    assert recall_at_k(
        ["api-framework", "other", "database"],
        ["api-framework", "database"],
        2,
    ) == 0.5


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(
        ["wrong", "api-framework", "database"],
        ["api-framework", "database"],
    ) == 0.5


def test_reciprocal_rank_is_zero_when_no_relevant_result_exists():
    assert reciprocal_rank(["wrong", "other"], ["api-framework"]) == 0.0


def test_mean_reciprocal_rank_averages_questions():
    assert mean_reciprocal_rank(
        [["api-framework", "wrong"], ["wrong", "database"]],
        [["api-framework"], ["database"]],
    ) == pytest.approx(0.75)


def test_recall_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        recall_at_k(["api-framework"], ["api-framework"], 0)


def test_metrics_reject_empty_relevance_labels():
    with pytest.raises(ValueError, match="must not be empty"):
        reciprocal_rank(["api-framework"], [])


def test_mrr_rejects_mismatched_question_counts():
    with pytest.raises(ValueError, match="equal lengths"):
        mean_reciprocal_rank([["api-framework"]], [])


@pytest.fixture
def valid_benchmark_data():
    corpus = [
        {"id": "c1", "text": "FastAPI serves the HTTP API."},
        {"id": "c2", "text": "PostgreSQL stores application records."},
    ]
    dataset = [
        {"id": "q1", "question": "Which API framework?", "relevant_chunk_ids": ["c1"]}
    ]
    return corpus, dataset


def test_benchmark_validation_accepts_valid_data(valid_benchmark_data):
    validate_benchmark_data(*valid_benchmark_data)


@pytest.mark.parametrize(
    ("corpus", "dataset", "message"),
    [
        ([{"id": "", "text": "body"}], [{"id": "q1", "question": "Q", "relevant_chunk_ids": ["c1"]}], "non-empty string id"),
        ([{"id": "c1", "text": "one"}, {"id": "c1", "text": "two"}], [{"id": "q1", "question": "Q", "relevant_chunk_ids": ["c1"]}], "Corpus chunk IDs must be unique"),
        ([{"id": "c1", "text": "body"}], [{"id": "", "question": "Q", "relevant_chunk_ids": ["c1"]}], "non-empty string id"),
        ([{"id": "c1", "text": "body"}], [{"id": "q1", "question": "  ", "relevant_chunk_ids": ["c1"]}], "must be non-empty"),
        ([{"id": "c1", "text": "body"}], [{"id": "q1", "question": "Q", "relevant_chunk_ids": []}], "non-empty list"),
        ([{"id": "c1", "text": "body"}], [{"id": "q1", "question": "Q", "relevant_chunk_ids": ["missing"]}], "unknown corpus chunk IDs"),
        ([{"id": "c1", "text": "body"}], [{"id": "q1", "question": "Q", "relevant_chunk_ids": ["c1"]}, {"id": "q1", "question": "Q2", "relevant_chunk_ids": ["c1"]}], "question IDs must be unique"),
    ],
)
def test_benchmark_validation_rejects_invalid_data(corpus, dataset, message):
    with pytest.raises(ValueError, match=message):
        validate_benchmark_data(corpus, dataset)


def test_benchmark_report_generation_is_model_free(tmp_path, valid_benchmark_data):
    corpus, dataset = valid_benchmark_data
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "nested" / "report.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    assert run_benchmark(
        corpus_path,
        dataset_path,
        report_path,
        use_models=False,
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["bm25"]["recall_at_3"] == 1.0
    assert report["bm25"]["mrr"] == 1.0
    assert report["gate"] == {"minimum_recall": 0.75, "passed": True}


def test_benchmark_report_generation_accepts_fake_model_components(
    tmp_path,
    valid_benchmark_data,
):
    corpus, dataset = valid_benchmark_data
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    class FakeVectorStore:
        def add_chunks(self, chunks):
            self.chunks = chunks

        def search(self, question, top_k=3):
            return self.chunks[:top_k]

    class FakeHybridRetriever:
        def __init__(self, store):
            self.vector_store = store
            self.chunks = []
            self.bm25 = None

        @staticmethod
        def _tokenize(text):
            return text.lower().split()

        def search(self, question, top_k=3, candidate_k=10):
            return self.vector_store.search(question, top_k)

    class FakeEmbeddingModel:
        def encode(self, texts):
            return [[1.0] for _ in texts]

    class FakeReranker:
        def rerank(self, question, documents, top_k=3):
            return documents[:top_k]

    vector_store = FakeVectorStore()
    vector_store.add_chunks(corpus)
    components = {
        "embedding_model": FakeEmbeddingModel(),
        "vector_store": vector_store,
        "reranker": FakeReranker(),
        "hybrid_retriever_factory": FakeHybridRetriever,
    }

    assert run_benchmark(
        corpus_path,
        dataset_path,
        report_path,
        model_components=components,
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"bm25", "semantic", "hybrid", "hybrid_reranked", "gate"}
    assert report["hybrid_reranked"]["recall_at_3"] == 1.0


def test_benchmark_rejects_invalid_data_before_report_write(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    corpus_path.write_text('[{"id":"c1","text":"body"}]', encoding="utf-8")
    dataset_path.write_text(
        '[{"id":"q1","question":" ","relevant_chunk_ids":["c1"]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be non-empty"):
        run_benchmark(corpus_path, dataset_path, report_path, use_models=False)
    assert not report_path.exists()
