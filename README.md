# Production RAG Pipeline with CI-Gated Evaluation

## Document-grounded evaluation

`evaluation/dataset.json` is the benchmark for the resume used when this dataset was authored. Each example has an `id`, `question`, `expected_answer`, and `expected_keywords`. Questions and expected values must be supported by the document under test. Do not add a reference `context`: context recall is calculated only from passages returned by the RAG pipeline.

Select a dataset that matches the PDF explicitly. For the resume benchmark:

```powershell
python -m evaluation.run_rag_evaluation "path\to\your-document.pdf" --dataset evaluation/dataset.json
```

For another document type, create a separate JSON dataset with the same schema, containing only questions and expected answers verified against that document, and pass its path with `--dataset`. The answer keyword and retrieved-context recall gates are each 0.75; an example passes only when both gates pass.

## CI retrieval benchmark

GitHub Actions runs the unit tests and a deterministic BM25 retrieval gate over the labeled fixture corpus in `evaluation/benchmark/`. The benchmark fails when mean Recall@3 is below 0.75. Run the same model-free gate locally with:

```powershell
python -m evaluation.benchmark.run_benchmark --skip-models
```

To compare BM25, semantic, hybrid RRF, and CrossEncoder reranking, omit `--skip-models`. That full benchmark loads the configured embedding and reranker models and writes `evaluation/benchmark/results.json`.
