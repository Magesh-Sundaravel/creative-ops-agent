from datetime import date

import pytest

from creative_ops.agents.coordinator import CoordinatorResult, Escalation
from creative_ops.loop.agent_loop import LoopResult, StopOutcome, TokenUsage
from evals.cases import EvalCase
from evals.run_evals import run_cases, select_cases


def eval_case(case_id: str, escalation: str | None) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": case_id,
            "brief": case_id,
            "today": "2031-10-01",
            "tags": [],
            "expected": {
                "briefs": [],
                "ambiguities": [],
                "escalation": escalation,
                "campaigns": [],
            },
        }
    )


def not_a_brief() -> CoordinatorResult:
    return CoordinatorResult(
        campaigns=[],
        escalation=Escalation(reason="not_a_brief", answer="amend", needs=[], ambiguities=[]),
        intakes=[],
        reviews=[],
        failed_runs=[],
        loop_result=LoopResult(
            text="",
            outcome=StopOutcome.COMPLETED,
            messages=[],
            usage=TokenUsage(input_tokens=30, output_tokens=7),
        ),
        state=None,
    )


async def test_run_cases_records_a_result_per_case_and_a_raising_run_as_failed(
    connect, fake_client
):
    seen: list[tuple[str, date]] = []

    async def coordinate(client, session, brief, today):
        seen.append((brief, today))
        if brief == "raises":
            raise RuntimeError("boom")
        return not_a_brief()

    cases = [
        eval_case("passes", "not_a_brief"),
        eval_case("fails", None),
        eval_case("raises", "not_a_brief"),
    ]

    report = await run_cases(
        cases, fake_client(), fake_client(), coordinate=coordinate, open_session=connect
    )

    by_id = {r.case_id: r for r in report.cases}
    assert sorted(seen) == sorted((c.id, date(2031, 10, 1)) for c in cases)
    assert [r.case_id for r in report.cases] == ["passes", "fails", "raises"]
    assert (by_id["passes"].passed, by_id["fails"].passed, by_id["raises"].passed) == (
        True,
        False,
        False,
    )
    assert by_id["passes"].usage == TokenUsage(input_tokens=30, output_tokens=7)
    assert by_id["raises"].error == "RuntimeError: boom"
    assert by_id["raises"].grades == []
    assert report.scores["outcome"].model_dump() == {"passed": 1, "total": 2}


def test_select_cases_with_an_id_keeps_only_that_case():
    cases = [eval_case("a", None), eval_case("b", None)]

    assert [c.id for c in select_cases(cases, "b")] == ["b"]
    assert select_cases(cases, None) == cases


def test_select_cases_with_an_unknown_id_raises():
    with pytest.raises(ValueError, match="nope"):
        select_cases([eval_case("a", None)], "nope")


async def test_run_cases_a_session_that_fails_to_open_fails_only_its_case(connect, fake_client):
    async def coordinate(client, session, brief, today):
        return not_a_brief()

    def open_session():
        if opened.pop(0) == "broken":
            raise OSError("server did not start")
        return connect()

    cases = [eval_case("broken", "not_a_brief"), eval_case("ok", "not_a_brief")]
    opened = [c.id for c in cases]

    report = await run_cases(
        cases, fake_client(), fake_client(), coordinate=coordinate, open_session=open_session
    )

    assert [(r.case_id, r.passed, r.error) for r in report.cases] == [
        ("broken", False, "OSError: server did not start"),
        ("ok", True, None),
    ]
