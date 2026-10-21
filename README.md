# HRM Copilot

An HR assistant that answers company-policy questions with cited, evaluated
retrieval-augmented generation (RAG), and later handles HR workflows (leave,
expenses, payroll) through role-protected agents.

> Status: under construction. Part A (RAG) is being built and evaluated first.

## Quick start

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your Groq API key
hrm --help
```

## Using it

```bash
hrm ingest                                   # index the policies (only changed docs are re-embedded)
hrm search "can I freelance on the side?"    # show the most relevant policy sections
hrm chunk-stats                              # chunk counts and token sizes per document
```

## Retrieval design

Every option was measured on 40 answerable questions (facts, paraphrases, multi-part)
plus 10 unanswerable ones; see [evals/RESULTS.md](evals/RESULTS.md).
A result counts only if it comes from the right document **and** contains the needed fact.

| Setup | hit@1 | hit@5 | MRR@10 | recall@5 | ms/query |
|---|---|---|---|---|---|
| dense (bge-small) | 0.70 | 0.88 | 0.783 | 0.85 | 10 |
| BM25 | 0.62 | 0.90 | 0.754 | 0.89 | 0 |
| hybrid (dense + BM25, RRF) | 0.65 | 0.93 | 0.778 | 0.91 | 8 |
| dense + rerank | 0.82 | 0.95 | 0.881 | 0.94 | 921 |
| **hybrid + rerank, top 20 (default)** | **0.82** | **0.97** | **0.894** | **0.95** | 875 |
| hybrid + rerank, top 10 | 0.80 | 0.95 | 0.869 | 0.93 | 401 |

**Chosen: hybrid retrieval + cross-encoder reranking of the top 20.**

- **Hybrid**, because dense and BM25 fail on different questions: dense understands
  paraphrases ("vacation" → PTO), BM25 nails exact terms ("age 26", "401k"). Reciprocal
  Rank Fusion combines their *ranks*, so their incomparable scores don't matter.
- **Reranking**, because RRF gets the right chunk *into* the top 5 but blurs the order at
  the top. A cross-encoder reads question and chunk together and fixes it:
  hit@1 0.65 → 0.82, and 39 of 40 answerable questions have the right chunk in the top 5.
- **Top 20 rather than 10**: paraphrased answers often start at ranks 11–20; cutting to 10
  halves latency but drops paraphrase hit@5 from 0.89 to 0.78.
- **Cost**: ~0.9 s per question on a laptop CPU, mostly the reranker. Acceptable next to
  the LLM answer step; `HRM_RETRIEVE_K=10` is the measured faster option.
- **Not covered by the score alone**: fused (RRF) scores look the same for answerable and
  unanswerable questions; reranker scores separate them much better (mean 2.0 vs −8.4) but
  still overlap, so the answer step must also be able to say "not covered".

**Known weak spot**: long sections that list many topics (e.g. *Conflicts of interest*, one
385-token chunk) score low with the reranker when the question matches only one item in the
list ("freelance work" → the "Outside employment" bullet); such answers land at rank 2–3
instead of 1. Smaller chunks for list-heavy sections are a candidate for a later, measured
improvement round.

Models (all local, no API calls): `BAAI/bge-small-en-v1.5` embeddings and
`Xenova/ms-marco-MiniLM-L-6-v2` reranker via fastembed; chunks of ≤ 400 tokens
(60 overlap, sections under 50 merged).

## Development

```bash
pre-commit install          # once: run ruff + mypy on every commit
pre-commit run --all-files  # lint, format and type-check everything
pytest                      # unit tests (no network, no LLM calls)
```

## Project layout

```
src/hrm/      application code (core config/logging/LLM, RAG pipeline, CLI)
tests/        unit tests (no network, no LLM calls)
data/docs/    HR policy documents
evals/        retrieval and answer evaluations (call the LLM; run manually)
```

## Data source

The policy documents in `data/docs/` come from Sourcegraph's public handbook
(obtained via a reference project). They are used here as realistic sample data.
