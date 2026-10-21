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
| 2026-10-21 | final default (hybrid + rerank top-20) | hybrid + rerank | rerank ms-marco-MiniLM-L-6-v2 top-20, RRF k=60, 20 candidates each, bge-small-en-v1.5, query prefix on, chunks 400/60/50 | 0.82 | 0.97 | 0.894 | 0.95 | 889 | 2.010 / -8.434 |
<!-- retrieval rows: new rows are inserted above this line -->

## Answers

Dataset: `evals/datasets/rag_answers.jsonl` (30 answerable + 10 unanswerable questions).
Run: `.venv/bin/python evals/run_answers.py --label "<what changed>"`.
Judged by an LLM (see `src/hrm/evaluation/answers.py`): correctness = share of key facts
present; fully correct = all key facts and no forbidden claim; faithfulness = no claim
unsupported by the documents; refusals are checked by code.

| date | label | config | correct | fully | faithful | cites | false ref. | ref. ok | s/q | tok/q |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | baseline (prompt v1) | hybrid + rerank, answer openai/gpt-oss-120b, prompt v1, judge openai/gpt-oss-20b | 0.69 | 0.67 | 0.95 | 1.00 | 0.23 | 0.90 | 6.2 | 1500 |
<!-- answer rows: new rows are inserted above this line -->
