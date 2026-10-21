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
| 2026-10-21 | hybrid RRF (dense+bm25) | hybrid | RRF k=60, 20 candidates each, bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.65 | 0.93 | 0.778 | 0.91 | 8 | 0.033 / 0.030 |
| 2026-10-21 | hybrid + rerank | hybrid + rerank | rerank ms-marco-MiniLM-L-6-v2 top-20, RRF k=60, 20 candidates each, bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.82 | 0.97 | 0.894 | 0.95 | 875 | 2.010 / -8.434 |
| 2026-10-21 | dense + rerank | dense + rerank | rerank ms-marco-MiniLM-L-6-v2 top-20, bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.82 | 0.95 | 0.881 | 0.94 | 921 | 2.010 / -8.738 |
| 2026-10-21 | hybrid + rerank top-10 | hybrid + rerank | rerank ms-marco-MiniLM-L-6-v2 top-10, RRF k=60, 10 candidates each, bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.80 | 0.95 | 0.869 | 0.93 | 401 | 2.003 / -8.441 |
<!-- retrieval rows: new rows are inserted above this line -->
