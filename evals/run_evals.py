"""Run the Eval Cases through the Coordinator: `uv run python evals/run_evals.py [--case <id>]`.

Each case runs once, against a fresh client stack, three at a time. Prints a pass/fail
table and scores per Grader and per field, and writes the full report as JSON under
`evals/results/`.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

# Run as a script, only `evals/` is on the path: add the repo root so `evals` imports.
sys.path.insert(0, str(Path(__file__).parents[1]))

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel

from creative_ops.agents.coordinator import CoordinatorResult, coordinate
from creative_ops.config import get_settings
from creative_ops.loop.agent_loop import TokenUsage
from creative_ops.tools.mcp_adapter import open_client_stack
from evals.cases import EvalCase, load_cases
from evals.graders import Grade, grade_case

logger = logging.getLogger(__name__)

CONCURRENCY = 3
RESULTS_DIR = Path(__file__).parent / "results"

Coordinate = Callable[[AsyncAnthropic, Client, str, date], Awaitable[CoordinatorResult]]


class CaseResult(BaseModel):
    """One Eval Case's Run, graded; a Run that raised has its error and no grades."""

    case_id: str
    passed: bool
    grades: list[Grade]
    usage: TokenUsage | None
    """Tokens across the Coordinator and every subagent Run; None if the Run raised."""
    error: str | None = None


class GraderScore(BaseModel):
    """Passed Grades out of all a Grader gave, pooled across cases."""

    passed: int
    total: int


class Report(BaseModel):
    """Every case's result and the scores pooled per Grader and per field."""

    cases: list[CaseResult]
    scores: dict[str, GraderScore]
    field_scores: dict[str, GraderScore]


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
    )


async def _run_case(
    case: EvalCase,
    client: AsyncAnthropic,
    judge_client: AsyncAnthropic,
    coordinate: Coordinate,
    open_session: Callable[[], Client],
) -> CaseResult:
    """Run one case on its own client-stack session and grade it; a raise fails the case."""
    try:
        async with open_session() as session:
            result = await coordinate(client, session, case.brief, case.today)
        grades = await grade_case(case, result, judge_client)
    except Exception as exc:  # boundary: one bad case must not end the eval
        logger.exception("Eval Case %r raised", case.id)
        return CaseResult(case_id=case.id, passed=False, grades=[], usage=None, error=_error(exc))
    return CaseResult(
        case_id=case.id, passed=grades.passed, grades=grades.grades, usage=_usage(result)
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

    results = await asyncio.gather(*(bounded(c) for c in cases))
    scores: dict[str, GraderScore] = {}
    field_scores: dict[str, GraderScore] = {}
    for grade in (g for r in results for g in r.grades):
        for pooled, key in [(scores, grade.grader), (field_scores, grade.field)]:
            score = pooled.setdefault(key, GraderScore(passed=0, total=0))
            score.passed += grade.passed
            score.total += 1
    return Report(cases=list(results), scores=scores, field_scores=field_scores)


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
    for grader, score in report.scores.items():
        print(f"{grader}: {score.passed}/{score.total}")
    print()
    for field, score in report.field_scores.items():
        print(f"  {field}: {score.passed}/{score.total}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected Eval Cases against the live API, print and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run only the Eval Case with this id.")
    args = parser.parse_args(argv)
    try:
        cases = select_cases(load_cases(), args.case)
    except ValueError as exc:
        parser.error(str(exc))
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    report = asyncio.run(run_cases(cases, client, client))
    _print_report(report)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(report.model_dump_json(indent=2))
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
