"""Run the Eval Cases through the Coordinator: `uv run python evals/run_evals.py [--case <id>]`.

Each case runs once, against a fresh client stack, three at a time. Prints a pass/fail
table and scores per Grader and per field, and writes the full report as JSON under
`evals/results/`. Exits 1 if any case's Run raised, no case ran, or the share of passed
cases is below `--min-pass-rate`.
"""

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

# Run as a script, only `evals/` is on the path: add the repo root so `evals` imports.
sys.path.insert(0, str(Path(__file__).parents[1]))

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel

from creative_ops import tracing
from creative_ops.agents.coordinator import CoordinatorResult, coordinate
from creative_ops.config import get_settings
from creative_ops.logs import configure_logging, current_brief_id
from creative_ops.loop.agent_loop import TokenUsage
from creative_ops.tools.mcp_adapter import open_client_stack
from evals.cases import EvalCase, load_cases
from evals.graders import Grade, grade_case

logger = logging.getLogger(__name__)

CONCURRENCY = 3
MIN_PASS_RATE = 0.9
RESULTS_DIR = Path(__file__).parent / "results"

Coordinate = Callable[[AsyncAnthropic, Client, str, date], Awaitable[CoordinatorResult]]


class CaseResult(BaseModel):
    """One Eval Case's Run, graded; a Run that raised has its error and no grades."""

    case_id: str
    passed: bool
    grades: list[Grade]
    usage: TokenUsage | None
    """Tokens and cost across the Coordinator and every subagent Run; None if the Run raised."""
    error: str | None = None
    latency_s: float = 0
    """Seconds the case took, Run and grading."""


class GraderScore(BaseModel):
    """Passed Grades out of all a Grader gave, pooled across cases."""

    passed: int
    total: int


class Totals(BaseModel):
    """Tokens and cost summed over the cases that completed, and wall-clock time.

    Cases that raised and Judge calls are not counted; cost is None if any is unknown.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None
    latency_s: float


class Report(BaseModel):
    """Every case's result, their totals, and the scores pooled per Grader and per field."""

    cases: list[CaseResult]
    totals: Totals
    scores: dict[str, GraderScore]
    field_scores: dict[str, GraderScore]


def _total_cost(usages: Sequence[TokenUsage]) -> Decimal | None:
    """The summed cost, or None if any is unknown."""
    if any(u.cost_usd is None for u in usages):
        return None
    return sum((u.cost_usd for u in usages), Decimal(0))


def _usage(result: CoordinatorResult) -> TokenUsage:
    """Tokens used by every Run recorded in `result`, failed ones included."""
    usages = [r.loop_result.usage for r in [*result.intakes, *result.reviews]]
    usages += [c.handoff.loop_result.usage for c in result.campaigns if c.handoff]
    usages += [f.usage for f in result.failed_runs]
    if result.loop_result is not None:
        usages.append(result.loop_result.usage)
    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=_total_cost(usages),
    )


async def _run_case(
    case: EvalCase,
    client: AsyncAnthropic,
    judge_client: AsyncAnthropic,
    coordinate: Coordinate,
    open_session: Callable[[], Client],
) -> CaseResult:
    """Run one case on its own client-stack session and grade it; a raise fails the case."""
    current_brief_id.set(case.id)  # this task's own context, so it never leaks to other cases
    start = time.monotonic()
    with tracing.trace_case(case.id, case.brief) as trace:
        try:
            async with open_session() as session:
                result = await coordinate(client, session, case.brief, case.today)
            grades = await grade_case(case, result, judge_client)
        except Exception as exc:  # boundary: one bad case must not end the eval
            logger.exception("Eval Case %r raised", case.id)
            error = _error(exc)
            trace.update(output=error, level="ERROR", status_message=error)
            return CaseResult(
                case_id=case.id,
                passed=False,
                grades=[],
                usage=None,
                error=error,
                latency_s=time.monotonic() - start,
            )
        trace.update(output="passed" if grades.passed else "failed")
    return CaseResult(
        case_id=case.id,
        passed=grades.passed,
        grades=grades.grades,
        usage=_usage(result),
        latency_s=time.monotonic() - start,
    )


def _error(exc: BaseException) -> str:
    """The exception as `Name: message`, unwrapped from groups (an MCP session's, the Judge's).

    A group of several is shown as its first exception.
    """
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


async def run_cases(
    cases: Sequence[EvalCase],
    client: AsyncAnthropic,
    judge_client: AsyncAnthropic,
    *,
    coordinate: Coordinate = coordinate,
    open_session: Callable[[], Client] = open_client_stack,
) -> Report:
    """Run and grade every case, at most `CONCURRENCY` at a time, results in case order."""
    limit = asyncio.Semaphore(CONCURRENCY)

    async def bounded(case: EvalCase) -> CaseResult:
        async with limit:
            return await _run_case(case, client, judge_client, coordinate, open_session)

    start = time.monotonic()
    results = await asyncio.gather(*(bounded(c) for c in cases))
    usages = [r.usage for r in results if r.usage is not None]
    totals = Totals(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=_total_cost(usages),
        latency_s=time.monotonic() - start,
    )
    scores: dict[str, GraderScore] = {}
    field_scores: dict[str, GraderScore] = {}
    for grade in (g for r in results for g in r.grades):
        for pooled, key in [(scores, grade.grader), (field_scores, grade.field)]:
            score = pooled.setdefault(key, GraderScore(passed=0, total=0))
            score.passed += grade.passed
            score.total += 1
    return Report(cases=list(results), totals=totals, scores=scores, field_scores=field_scores)


def select_cases(cases: list[EvalCase], case_id: str | None) -> list[EvalCase]:
    """Only the case with `case_id`, or every case when it is None."""
    if case_id is None:
        return cases
    selected = [c for c in cases if c.id == case_id]
    if not selected:
        raise ValueError(f"No Eval Case {case_id!r}; ids are {[c.id for c in cases]}")
    return selected


def _print_report(report: Report) -> None:
    """A pass/fail line per case with its failed Grades, then scores per Grader and per field."""
    for case in report.cases:
        print(f"{'PASS' if case.passed else 'FAIL'}  {case.case_id}")
        if case.error:
            print(f"      error: {case.error}")
        for g in (g for g in case.grades if not g.passed):
            where = f"{g.campaign}: " if g.campaign else ""
            print(f"      {g.grader} {where}{g.field}: {g.reason}")
    passed = sum(c.passed for c in report.cases)
    print(f"\nCases: {passed}/{len(report.cases)} passed")
    t = report.totals
    cost = "n/a" if t.cost_usd is None else f"${t.cost_usd:.4f}"
    print(f"Tokens: {t.input_tokens} in, {t.output_tokens} out; cost {cost}; {t.latency_s:.1f}s")
    for grader, score in report.scores.items():
        print(f"{grader}: {score.passed}/{score.total}")
    print()
    for field, score in report.field_scores.items():
        print(f"  {field}: {score.passed}/{score.total}")


def gate_failure(report: Report, min_pass_rate: float) -> str | None:
    """Why the eval fails CI, or None if it passes.

    It fails on a case whose Run raised, an empty case set, or a pass rate below the threshold.
    """
    if not report.cases:
        return "no Eval Cases ran"
    errored = sum(c.error is not None for c in report.cases)
    if errored:
        return f"{errored} case(s) raised"
    rate = sum(c.passed for c in report.cases) / len(report.cases)
    if rate < min_pass_rate:
        return f"pass rate {rate:.2f} is below {min_pass_rate:.2f}"
    return None


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected Eval Cases against the live API, print and save the report.

    Exits 1 per `gate_failure`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run only the Eval Case with this id.")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=MIN_PASS_RATE,
        help="Exit 1 if the share of passed cases is below this (default %(default)s).",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.min_pass_rate <= 1:
        parser.error("--min-pass-rate must be between 0 and 1")
    try:
        cases = select_cases(load_cases(), args.case)
    except ValueError as exc:
        parser.error(str(exc))
    settings = get_settings()
    configure_logging(settings.log_level)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    report = asyncio.run(run_cases(cases, client, client))
    tracing.flush()
    _print_report(report)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(report.model_dump_json(indent=2))
    print(f"\nReport: {path}")
    failure = gate_failure(report, args.min_pass_rate)
    if failure:
        print(f"FAILED: {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
