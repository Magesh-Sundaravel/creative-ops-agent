"""Graders: score one Run's result against an Eval Case (code checks the facts, ADR 0003)."""

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from creative_ops.agents.coordinator import CampaignOutcome, CoordinatorResult
from evals.cases import EvalCase, ExpectedCampaign


class Grade(BaseModel):
    """One Grader's pass/fail on one field, of one campaign or (campaign None) the Brief."""

    grader: str
    campaign: str | None
    field: str
    passed: bool
    reason: str | None = None
    """Why it failed: expected and produced values, or the Judge's reason."""


class CaseGrades(BaseModel):
    """Every Grade for one Eval Case."""

    case_id: str
    grades: list[Grade]

    @property
    def passed(self) -> bool:
        """The case passes only if every Grade does."""
        return all(g.passed for g in self.grades)


def _key(brand: str | None, campaign_name: str | None) -> tuple[str, str]:
    """A campaign's pairing key: brand and campaign name, lower-cased and stripped."""
    return ((brand or "").strip().lower(), (campaign_name or "").strip().lower())


def _produced_outcome(outcome: CampaignOutcome) -> str:
    """What a campaign ended as, in the terms of `ExpectedCampaign.outcome` and its reason."""
    if outcome.handoff is not None:
        return "handed_off"
    if outcome.escalation is None:
        return "unfinished"
    if outcome.escalation.reason == "revisions_requested":
        return "revisions_requested"
    return f"escalated ({outcome.escalation.reason})"


def _expected_outcome(expected: ExpectedCampaign) -> str:
    """`expected`'s outcome, in the same terms as `_produced_outcome`."""
    return f"escalated ({expected.reason})" if expected.outcome == "escalated" else expected.outcome


def _grade(grader: str, campaign: str | None, field: str, expected: object, got: object) -> Grade:
    """A Grade that passes when `got` equals `expected`, saying both when it fails."""
    passed = expected == got
    reason = None if passed else f"expected {expected!r}, got {got!r}"
    return Grade(grader=grader, campaign=campaign, field=field, passed=passed, reason=reason)


def _outcome_grades(case: EvalCase, result: CoordinatorResult) -> list[Grade]:
    """The Brief-level Escalation reason, and each paired campaign's outcome."""
    produced_escalation = result.escalation.reason if result.escalation else None
    grades = [
        _grade("outcome", None, "brief_escalation", case.expected.escalation, produced_escalation)
    ]
    produced = {_key(c.brand, c.campaign_name): c for c in result.campaigns}
    for expected in case.expected.campaigns:
        match = produced.get(_key(expected.brand, expected.campaign_name))
        if match is not None:
            grades.append(
                _grade(
                    "outcome",
                    f"{expected.brand} / {expected.campaign_name}",
                    "campaign_outcome",
                    _expected_outcome(expected),
                    _produced_outcome(match),
                )
            )
    return grades


async def grade_case(
    case: EvalCase, result: CoordinatorResult, judge_client: AsyncAnthropic
) -> CaseGrades:
    """Every Grade for one Run's result against its Eval Case."""
    return CaseGrades(case_id=case.id, grades=_outcome_grades(case, result))
