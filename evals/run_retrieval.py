"""Evaluate retrieval on evals/datasets/rag_retrieval.jsonl and record a row in RESULTS.md.

Uses the real index (run `hrm ingest` first) and the real embedding model.

    .venv/bin/python evals/run_retrieval.py --label "dense baseline"
    .venv/bin/python evals/run_retrieval.py --retriever hybrid --label "hybrid RRF"
    .venv/bin/python evals/run_retrieval.py --retriever hybrid --rerank --label "hybrid + rerank"
    .venv/bin/python evals/run_retrieval.py --no-record      # just print the report
"""

import argparse
import time
from datetime import date
from pathlib import Path
from statistics import mean

from hrm.core.config import get_settings
from hrm.core.logging import configure_logging
from hrm.evaluation.retrieval import (
    MRR_DEPTH,
    Metrics,
    QuestionResult,
    breakdown,
    load_dataset,
    metrics,
)
from hrm.rag.embeddings import FastEmbedEmbedder
from hrm.rag.ingest import Manifest
from hrm.rag.retrieval.factory import build_retriever
from hrm.rag.vectorstore import ChromaStore

EVALS_DIR = Path(__file__).parent
DATASET = EVALS_DIR / "datasets" / "rag_retrieval.jsonl"
RESULTS = EVALS_DIR / "RESULTS.md"
ROWS_MARKER = "<!-- retrieval rows: new rows are inserted above this line -->"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        help="Overrides HRM_RETRIEVAL_MODE for this run.",
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank candidates with the cross-encoder."
    )
    parser.add_argument("--label", default="", help="Short name for this run in RESULTS.md.")
    parser.add_argument("--no-record", action="store_true", help="Do not write RESULTS.md.")
    args = parser.parse_args()

    settings = get_settings()
    if args.retriever:
        settings = settings.model_copy(update={"retrieval_mode": args.retriever})
    if args.rerank:
        settings = settings.model_copy(update={"rerank": True})
    mode = settings.retrieval_mode + (" + rerank" if settings.rerank else "")
    configure_logging("WARNING", settings.log_format)  # keep per-query logs out of the report
    _check_index_matches_settings(settings.index_dir / "manifest.json", settings.embedding_model)
    retriever = build_retriever(
        settings, FastEmbedEmbedder.from_settings(settings), ChromaStore.from_settings(settings)
    )

    results = []
    for question in load_dataset(DATASET):
        start = time.perf_counter()
        retrieved = retriever.retrieve(question.question, MRR_DEPTH)
        results.append(QuestionResult(question, retrieved, (time.perf_counter() - start) * 1000))

    overall = metrics(results)
    _print_report(results, overall)

    chunks = f"chunks {settings.chunk_size}/{settings.chunk_overlap}/{settings.chunk_min_size}"
    embedding = (
        f"{settings.embedding_model.split('/')[-1]}, "
        f"query prefix {'on' if settings.embedding_query_prefix else 'off'}"
    )
    configs = {
        "dense": f"{embedding}, {chunks}",
        "bm25": chunks,
        "hybrid": f"RRF k={settings.rrf_k}, {settings.retrieve_k} candidates each, "
        f"{embedding}, {chunks}",
    }
    config = configs[settings.retrieval_mode]
    if settings.rerank:
        config = (
            f"rerank {settings.reranker_model.split('/')[-1]} top-{settings.retrieve_k}, {config}"
        )
    if not args.no_record:
        _record(
            results,
            overall,
            label=args.label or mode,
            retriever=mode,
            config=config,
        )
        print(f"\nRecorded in {RESULTS.relative_to(EVALS_DIR.parent)}")


def _check_index_matches_settings(manifest_path: Path, embedding_model: str) -> None:
    if not manifest_path.exists():
        raise SystemExit("No index found. Run `hrm ingest` first.")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.settings.embedding_model != embedding_model:
        raise SystemExit(
            f"Index was built with {manifest.settings.embedding_model}, settings say "
            f"{embedding_model}. Run `hrm ingest` first."
        )


def _print_report(results: list[QuestionResult], overall: Metrics) -> None:
    print(f"Answerable questions: {overall.n}")
    print(_metrics_header())
    print(_metrics_line("ALL", overall))
    for title, by in [("by question type", "type"), ("by document", "doc")]:
        print(f"\n{title}:")
        for key, m in breakdown(results, by=by).items():  # type: ignore[arg-type]
            print(_metrics_line(key, m))

    answerable, unanswerable = _top_scores(results)
    print(
        f"\nTop-1 score  answerable: mean {mean(answerable):.3f}, min {min(answerable):.3f}"
        f"  |  unanswerable: mean {mean(unanswerable):.3f}, max {max(unanswerable):.3f}"
    )
    print(f"Latency: {mean(r.ms for r in results):.0f} ms/query")

    failures = [r for r in results if r.question.answerable and not r.hit_at(5)]
    print(f"\nNot found in top 5 ({len(failures)}):")
    for r in failures:
        expected = ", ".join(f"{e.doc_id} > {e.section}" for e in r.question.expected)
        got = ", ".join(x.chunk.chunk_id for x in r.results[:3])
        print(f"  {r.question.id} [{r.question.type}] {r.question.question}")
        print(f"      expected: {expected}\n      got:      {got}")


def _metrics_header() -> str:
    return f"{'':<26} {'n':>3} {'hit@1':>6} {'hit@5':>6} {'MRR@10':>7} {'recall@5':>9}"


def _metrics_line(name: str, m: Metrics) -> str:
    return (
        f"{name:<26} {m.n:>3} {m.hit_at_1:>6.2f} {m.hit_at_5:>6.2f} {m.mrr:>7.3f} "
        f"{m.recall_at_5:>9.2f}"
    )


def _top_scores(results: list[QuestionResult]) -> tuple[list[float], list[float]]:
    answerable = [r.top_score for r in results if r.question.answerable and r.top_score is not None]
    unanswerable = [
        r.top_score for r in results if not r.question.answerable and r.top_score is not None
    ]
    return answerable, unanswerable


def _record(
    results: list[QuestionResult], m: Metrics, *, label: str, retriever: str, config: str
) -> None:
    if not RESULTS.exists():
        RESULTS.write_text(_results_template(), encoding="utf-8")
    answerable, unanswerable = _top_scores(results)
    row = (
        f"| {date.today().isoformat()} | {label} | {retriever} | {config} | {m.hit_at_1:.2f} | "
        f"{m.hit_at_5:.2f} | {m.mrr:.3f} | {m.recall_at_5:.2f} | "
        f"{mean(r.ms for r in results):.0f} | {mean(answerable):.3f} / {mean(unanswerable):.3f} |"
    )
    text = RESULTS.read_text(encoding="utf-8")
    RESULTS.write_text(text.replace(ROWS_MARKER, f"{row}\n{ROWS_MARKER}"), encoding="utf-8")


def _results_template() -> str:
    return f"""# Evaluation results

Every retrieval or generation change adds a row here, so improvements are measured, not guessed.

## Retrieval

Dataset: `evals/datasets/rag_retrieval.jsonl` (40 answerable + 10 unanswerable questions).
Run: `.venv/bin/python evals/run_retrieval.py --label "<what changed>"`.
A result is relevant when it comes from the expected document and contains the evidence phrase.
The last column compares the mean top-1 score of answerable vs. unanswerable questions.

| date | label | retriever | config | hit@1 | hit@5 | MRR@10 | recall@5 | ms/q | top-1 ans/unans |
|---|---|---|---|---|---|---|---|---|---|
{ROWS_MARKER}
"""


if __name__ == "__main__":
    main()
