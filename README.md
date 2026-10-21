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
