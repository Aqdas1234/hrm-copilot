"""Evaluate answers end to end (retrieve -> context -> answer) with an LLM judge.

Calls Groq: about 2 calls per question. The free tier allows 8,000 tokens per minute per
model, so a full run takes several minutes; rate-limit errors are retried automatically.

    .venv/bin/python evals/run_answers.py --label "baseline"
    .venv/bin/python evals/run_answers.py --ids a03,a30 --no-record   # a few questions
"""

import argparse
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel

from hrm.core.config import get_settings
from hrm.core.errors import LLMError
from hrm.core.llm import get_fast_model
from hrm.core.logging import configure_logging
from hrm.evaluation.answers import (
    AnswerCase,
    AnswerMetrics,
    CaseResult,
    Judge,
    JudgeVerdict,
    answer_metrics,
    load_cases,
)
from hrm.rag.generation import PROMPT_VERSION, not_covered
from hrm.rag.service import RAGService

EVALS_DIR = Path(__file__).parent
DATASET = EVALS_DIR / "datasets" / "rag_answers.jsonl"
RESULTS = EVALS_DIR / "RESULTS.md"
RUNS_DIR = EVALS_DIR / "runs"
ROWS_MARKER = "<!-- answer rows: new rows are inserted above this line -->"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="", help="Short name for this run in RESULTS.md.")
    parser.add_argument("--ids", default="", help="Comma-separated case ids to run (default all).")
    parser.add_argument("--no-record", action="store_true", help="Do not write RESULTS.md.")
    args = parser.parse_args()

    # Patient retries: the free tier's tokens-per-minute limit is hit during a full run.
    settings = get_settings().model_copy(update={"llm_max_retries": 8, "llm_timeout_s": 120.0})
    configure_logging("WARNING", settings.log_format)
    service = RAGService.from_settings(settings)  # the same pipeline as `hrm ask` and the API
    judge = Judge(get_fast_model(settings).with_structured_output(JudgeVerdict))

    cases = load_cases(DATASET)
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.id in wanted]

    results: list[CaseResult] = []
    for n, case in enumerate(cases, start=1):
        print(f"[{n}/{len(cases)}] {case.id} {case.question}", flush=True)
        start = time.perf_counter()
        try:
            with get_usage_metadata_callback() as usage:
                result = service.answer(case.question)
        except LLMError as exc:
            results.append(_failed(case, f"answer: {exc}", time.perf_counter() - start))
            continue
        answer, context = result.answer, result.context
        seconds = time.perf_counter() - start
        tokens = sum(u["total_tokens"] for u in usage.usage_metadata.values())

        verdict, error = None, None
        if case.answerable and answer.found:
            try:
                verdict = judge.judge(case, answer, context)
            except LLMError as exc:
                error = f"judge: {exc}"
        results.append(CaseResult(case, answer, verdict, seconds, tokens, error))

    overall = answer_metrics(results)
    _print_report(results, overall)
    run_file = _save_run(results)
    print(f"\nPer-question details: {run_file.relative_to(EVALS_DIR.parent)}")

    config = (
        f"{settings.retrieval_mode}{' + rerank' if settings.rerank else ''}, "
        f"answer {settings.answer_model}, prompt v{PROMPT_VERSION}, judge {settings.fast_model}"
    )
    failed = [r for r in results if r.error]
    if failed:
        # Errors (e.g. a daily rate limit) would be scored as refusals and distort the metrics.
        print(f"\n{len(failed)} question(s) failed with errors; NOT recorded in RESULTS.md.")
        raise SystemExit(1)
    if not args.no_record and not args.ids:
        _record(overall, label=args.label or "answers", config=config)
        print(f"Recorded in {RESULTS.relative_to(EVALS_DIR.parent)}")


def _failed(case: AnswerCase, error: str, seconds: float) -> CaseResult:
    return CaseResult(case, not_covered(), None, seconds, 0, error)


def _print_report(results: list[CaseResult], m: AnswerMetrics) -> None:
    print(f"\nAnswerable: {m.answerable}, unanswerable: {len(results) - m.answerable}")
    print(f"  correctness (key facts present)   {m.correctness:.2f}")
    print(f"  fully correct (all facts, no forbidden claim) {m.fully_correct:.2f}")
    print(f"  faithfulness (no unsupported claim)  {m.faithfulness:.2f}")
    print(f"  citation accuracy                 {m.citation_accuracy:.2f}")
    print(f"  false refusals (answerable)       {m.false_refusals:.2f}")
    print(f"  refusal accuracy (unanswerable)   {m.refusal_accuracy:.2f}")
    print(f"  per question: {m.seconds:.1f} s, {m.tokens:.0f} answer tokens")

    by_type: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_type[r.case.type].append(r)
    print("\nby type:  correctness / fully correct / faithful")
    for qtype, group in sorted(by_type.items()):
        if qtype != "unanswerable":
            g = answer_metrics(group)
            print(
                f"  {qtype:<11} {g.correctness:.2f} / {g.fully_correct:.2f} / {g.faithfulness:.2f}"
            )

    problems = [r for r in results if _problems(r)]
    print(f"\nQuestions with problems ({len(problems)}):")
    for r in problems:
        print(f"  {r.case.id} [{r.case.type}] {r.case.question}")
        print(f"      answer: {' '.join(r.answer.answer.split())[:300]}")
        for line in _problems(r):
            print(f"      - {line}")


def _problems(r: CaseResult) -> list[str]:
    lines = []
    if r.error:
        lines.append(f"ERROR {r.error}")
    if r.case.answerable and r.refused and not r.error:
        lines.append("refused an answerable question")
    if not r.case.answerable and not r.refused:
        lines.append("answered an unanswerable question")
    if r.verdict:
        lines += [f"missing fact: {c.statement}" for c in r.verdict.key_facts if not c.holds]
        lines += [f"FORBIDDEN claim: {s}" for s in r.forbidden_asserted]
        lines += [f"unsupported: {c}" for c in r.verdict.unsupported_claims]
        lines += [f"miscited: {c}" for c in r.verdict.miscited_claims]
    return lines


def _save_run(results: list[CaseResult]) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"answers-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                CaseRecord(
                    id=r.case.id,
                    question=r.case.question,
                    answer=r.answer.answer,
                    found=r.answer.found,
                    citations=[c.chunk_id for c in r.answer.citations],
                    verdict=r.verdict,
                    problems=_problems(r),
                    seconds=round(r.seconds, 2),
                    tokens=r.tokens,
                ).model_dump_json()
                + "\n"
            )
    return path


def _record(m: AnswerMetrics, *, label: str, config: str) -> None:
    text = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else "# Evaluation results\n"
    if ROWS_MARKER not in text:
        text = text.rstrip("\n") + "\n\n" + _answers_section()
    row = (
        f"| {date.today().isoformat()} | {label} | {config} | {m.correctness:.2f} | "
        f"{m.fully_correct:.2f} | {m.faithfulness:.2f} | {m.citation_accuracy:.2f} | "
        f"{m.false_refusals:.2f} | {m.refusal_accuracy:.2f} | {m.seconds:.1f} | {m.tokens:.0f} |"
    )
    RESULTS.write_text(text.replace(ROWS_MARKER, f"{row}\n{ROWS_MARKER}"), encoding="utf-8")


def _answers_section() -> str:
    return f"""## Answers

Dataset: `evals/datasets/rag_answers.jsonl` (31 answerable + 10 unanswerable questions).
Run: `.venv/bin/python evals/run_answers.py --label "<what changed>"`.
Judged by an LLM (see `src/hrm/evaluation/answers.py`): correctness = share of key facts
present; fully correct = all key facts and no forbidden claim; faithfulness = no claim
unsupported by the documents; refusals are checked by code.

| date | label | config | correct | fully | faithful | cites | false ref. | ref. ok | s/q | tok/q |
|---|---|---|---|---|---|---|---|---|---|---|
{ROWS_MARKER}
"""


class CaseRecord(BaseModel):
    """One line of the per-question run file."""

    id: str
    question: str
    answer: str
    found: bool
    citations: list[str]
    verdict: JudgeVerdict | None
    problems: list[str]
    seconds: float
    tokens: int


if __name__ == "__main__":
    main()
