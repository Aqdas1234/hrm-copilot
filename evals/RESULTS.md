# Evaluation results

Every retrieval or generation change adds a row here, so improvements are measured, not guessed.

## Retrieval

Dataset: `evals/datasets/rag_retrieval.jsonl` (40 answerable + 10 unanswerable questions).
Run: `.venv/bin/python evals/run_retrieval.py --label "<what changed>"`.
A result is relevant when it comes from the expected document and contains the evidence phrase.
The last column compares the mean top-1 score of answerable vs. unanswerable questions.

| date | label | retriever | config | hit@1 | hit@5 | MRR@10 | recall@5 | ms/q | top-1 ans/unans |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | dense baseline | dense | bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.70 | 0.88 | 0.783 | 0.85 | 10 | 0.734 / 0.599 |
| 2026-10-21 | bm25 only | bm25 | chunks 400/60/50 | 0.62 | 0.90 | 0.754 | 0.89 | 0 | 12.501 / 5.735 |
<!-- retrieval rows: new rows are inserted above this line -->
