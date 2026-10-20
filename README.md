# Production RAG Pipeline with CI-Gated Evaluation

## Document-grounded evaluation

`evaluation/dataset.json` is the benchmark for the resume used when this dataset was authored. Each example has an `id`, `question`, `expected_answer`, and `expected_keywords`. Questions and expected values must be supported by the document under test. Do not add a reference `context`: context recall is calculated only from passages returned by the RAG pipeline.

Select a dataset that matches the PDF explicitly. For the resume benchmark:

```powershell
python -m evaluation.run_rag_evaluation "$HOME\Downloads\Chethana_Keshava_Shettigar_Resume (1).pdf" --dataset evaluation/dataset.json
```

For another document type, create a separate JSON dataset with the same schema, containing only questions and expected answers verified against that document, and pass its path with `--dataset`. The answer keyword and retrieved-context recall gates are each 0.75; an example passes only when both gates pass.
