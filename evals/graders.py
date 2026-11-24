"""Graders: score one Run's result against an Eval Case (code checks the facts, ADR 0003)."""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Sequence

from anthropic import AsyncAnthropic
from anthropic.types import Message, ToolParam
from pydantic import BaseModel, Field

from creative_ops.agents.coordinator import CampaignOutcome, CoordinatorResult
from creative_ops.config import get_settings
from creative_ops.schemas.brief import Budget, Deliverable, StructuredBrief
from evals.cases import EvalCase, ExpectedCampaign, campaign_key

logger = logging.getLogger(__name__)


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


def _pair[E, P](
    expected: Sequence[E], produced: Sequence[P], key: Callable[[E | P], object]
) -> tuple[list[tuple[E, P]], list[E], list[P]]:
    """Pair each expected item with the first unpaired produced one of the same key.

    Returns the pairs, the missed expected items and the spurious produced ones; a second
    produced item with an already-paired key is spurious.
    """
    unpaired = list(produced)
    pairs: list[tuple[E, P]] = []
    missed: list[E] = []
    for e in expected:
        match = next((p for p in unpaired if key(p) == key(e)), None)
        if match is None:
            missed.append(e)
        else:
            unpaired.remove(match)
            pairs.append((e, match))
    return pairs, missed, unpaired


def _campaign_key(c: ExpectedCampaign | CampaignOutcome | StructuredBrief) -> tuple[str, str]:
    """`campaign_key` of an expected or produced campaign or Structured Brief."""
    return campaign_key(c.brand, c.campaign_name)


def _label(c: ExpectedCampaign | CampaignOutcome | StructuredBrief) -> str:
    """A campaign as `brand / campaign name`, for Grades and reasons."""
    return f"{c.brand} / {c.campaign_name}"


def _count_grade(missed: list[ExpectedCampaign], spurious: list[CampaignOutcome]) -> Grade:
    """Passes when every expected campaign paired and no produced one is left over."""
    problems = [f"missed {_label(c)!r}" for c in missed]
    problems += [f"spurious {_label(c)!r}" for c in spurious]
    return Grade(
        grader="pairing",
        campaign=None,
        field="campaign_count",
        passed=not problems,
        reason="; ".join(problems) or None,
    )


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
    """The Brief-level Escalation reason, the campaign count, and each paired campaign's outcome."""
    produced_escalation = result.escalation.reason if result.escalation else None
    pairs, missed, spurious = _pair(case.expected.campaigns, result.campaigns, _campaign_key)
    grades = [
        _grade("outcome", None, "brief_escalation", case.expected.escalation, produced_escalation),
        _count_grade(missed, spurious),
    ]
    for expected, match in pairs:
        grades.append(
            _grade(
                "outcome",
                _label(expected),
                "campaign_outcome",
                _expected_outcome(expected),
                _produced_outcome(match),
            )
        )
    return grades


def _norm(value: str | None) -> str:
    """A string lower-cased and stripped; None as empty."""
    return (value or "").strip().lower()


def _strip(value: str | None) -> str:
    """A string stripped but case kept; None as empty."""
    return (value or "").strip()


def _budget(budget: Budget | None) -> tuple[object, str] | None:
    """A Budget as (amount, normalised currency), so 8000 equals 8000.00 and gbp equals GBP.

    An empty Budget is no budget.
    """
    if budget is None or (budget.amount is None and budget.currency is None):
        return None
    return (budget.amount, _norm(budget.currency))


def _strings(values: list[str]) -> list[str]:
    """The distinct normalised strings, sorted, for comparing as a set."""
    return sorted({_norm(v) for v in values})


def _deliverable_key(d: Deliverable) -> tuple[str, str, int | None]:
    """What makes two Deliverables the same piece of work: asset type, channel, quantity."""
    return (_norm(d.asset_type), _norm(d.channel), d.quantity)


def _deliverable_grades(
    campaign: str, expected: StructuredBrief, got: StructuredBrief
) -> list[Grade]:
    """Deliverables as a multiset on `_deliverable_key`, then dimensions and due date per pair.

    Same-key Deliverables pair greedily on matching dimensions and due date, then on either,
    then on the key alone. Both lists are sorted first, so their order never changes a grade.
    """
    keys: list[Callable[[Deliverable], object]] = [
        lambda d: (*_deliverable_key(d), _norm(d.format_or_dimensions), d.due_date),
        lambda d: (*_deliverable_key(d), _norm(d.format_or_dimensions)),
        lambda d: (*_deliverable_key(d), d.due_date),
        _deliverable_key,
    ]
    pairs: list[tuple[Deliverable, Deliverable]] = []
    missed, spurious = sorted(expected.deliverables, key=repr), sorted(got.deliverables, key=repr)
    for key in keys:
        paired, missed, spurious = _pair(missed, spurious, key)
        pairs += paired
    problems = [f"missed {_deliverable_key(d)}" for d in missed]
    problems += [f"spurious {_deliverable_key(d)}" for d in spurious]
    grades = [
        Grade(
            grader="fields",
            campaign=campaign,
            field="deliverables",
            passed=not problems,
            reason="; ".join(problems) or None,
        )
    ]
    for e, g in pairs:
        grades += [
            _grade(
                "fields",
                campaign,
                "deliverable_dimensions",
                _norm(e.format_or_dimensions),
                _norm(g.format_or_dimensions),
            ),
            _grade("fields", campaign, "deliverable_due_date", e.due_date, g.due_date),
        ]
    return grades


def _brief_pairs(
    case: EvalCase, result: CoordinatorResult
) -> list[tuple[StructuredBrief, StructuredBrief]]:
    """Each expected Structured Brief with the last intake's brief of its campaign.

    Only a campaign whose key appears once on each side is paired. A missing one fails the
    count check; for a repeated key, which brief pairs with which is ambiguous, so none is.
    """
    produced = [c.brief for c in result.intakes[-1].campaigns] if result.intakes else []
    expected_counts = Counter(map(_campaign_key, case.expected.briefs))
    produced_by_key = {_campaign_key(b): b for b in produced}
    produced_counts = Counter(map(_campaign_key, produced))
    return [
        (e, produced_by_key[_campaign_key(e)])
        for e in case.expected.briefs
        if expected_counts[_campaign_key(e)] == 1 and produced_counts[_campaign_key(e)] == 1
    ]


def _field_grades(pairs: list[tuple[StructuredBrief, StructuredBrief]]) -> list[Grade]:
    """Each paired Structured Brief's factual fields."""
    grades: list[Grade] = []
    for e, g in pairs:
        campaign = _label(e)
        checks: list[tuple[str, object, object]] = [
            ("brand", _strip(e.brand), _strip(g.brand)),
            ("campaign_name", _strip(e.campaign_name), _strip(g.campaign_name)),
            ("season", _norm(e.season), _norm(g.season)),
            ("deadline", e.deadline, g.deadline),
            ("budget", _budget(e.budget), _budget(g.budget)),
            ("mandatories", _strings(e.mandatories), _strings(g.mandatories)),
            ("references", _strings(e.references), _strings(g.references)),
        ]
        grades += [_grade("fields", campaign, f, want, got) for f, want, got in checks]
        grades += _deliverable_grades(campaign, e, g)
    return grades


class Verdict(BaseModel):
    """Input of the Judge's `submit_verdict` tool."""

    passed: bool
    reason: str = Field(description="One line saying why.")


JUDGE_SYSTEM = """\
You are grading an AI agent's output against the expected output. Follow the rubric, then \
record your verdict by calling `submit_verdict` with whether it passes and a one-line reason."""

FIELD_RUBRIC = """\
Pass if the produced text means the same as the expected text: wording may differ, but every \
fact in the expected text is there and none is changed or added."""

AMBIGUITIES_RUBRIC = """\
Each list is a Brief's Ambiguities: things in the Brief that are unclear or contradictory. Pass \
only if every expected Ambiguity is covered by a produced one, however worded, and no produced \
Ambiguity is invented (covered by no expected one)."""

_VERDICT_TOOL: ToolParam = {
    "name": "submit_verdict",
    "description": "Record whether the produced output passes the rubric, with a one-line reason.",
    "input_schema": Verdict.model_json_schema(),
}


async def _judge(client: AsyncAnthropic, rubric: str, expected: str, produced: str) -> Message:
    """One Judge call on the fast model, forced to answer through `submit_verdict`."""
    return await client.messages.create(
        model=get_settings().model_fast,
        max_tokens=512,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<rubric>\n{rubric}\n</rubric>\n\n<expected>\n{expected}\n</expected>\n\n"
                    f"<produced>\n{produced}\n</produced>"
                ),
            }
        ],
        tools=[_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )


def _verdict(response: Message) -> Verdict:
    """The Judge's verdict: the input of its one `submit_verdict` call.

    Raises ValueError if the reply has no single tool_use or its input is not a Verdict.
    """
    [tool_use] = [b for b in response.content if b.type == "tool_use"]
    return Verdict.model_validate(tool_use.input)


async def _judge_grade(
    client: AsyncAnthropic,
    campaign: str | None,
    field: str,
    rubric: str,
    expected: str | None,
    produced: str | None,
) -> Grade:
    """A Judge Grade; a match or an empty side is decided in code, with no Judge call.

    A malformed Judge reply fails only this Grade, with the error as its reason; an API
    error propagates, so the case fails loudly rather than scoring a broken setup.
    """
    if _norm(expected) == _norm(produced):
        return Grade(grader="judge", campaign=campaign, field=field, passed=True)
    if not _norm(expected) or not _norm(produced):
        return _grade("judge", campaign, field, expected, produced)
    response = await _judge(client, rubric, expected or "", produced or "")
    try:
        verdict = _verdict(response)
    except ValueError as exc:
        logger.exception("Judge reply for %s %s is malformed", campaign or "Brief", field)
        return Grade(
            grader="judge",
            campaign=campaign,
            field=field,
            passed=False,
            reason=f"Judge failed: {type(exc).__name__}: {exc}",
        )
    return Grade(
        grader="judge",
        campaign=campaign,
        field=field,
        passed=verdict.passed,
        reason=verdict.reason,
    )


def _bullets(items: list[str]) -> str:
    """A list as one `- item` line each, for the Judge; empty for no items."""
    return "\n".join(f"- {item}" for item in items)


async def _judge_grades(
    case: EvalCase,
    result: CoordinatorResult,
    client: AsyncAnthropic,
    pairs: list[tuple[StructuredBrief, StructuredBrief]],
) -> list[Grade]:
    """Each paired brief's free-text fields, then the Brief's Ambiguities, judged concurrently.

    Ambiguities are judged only when the case expects campaigns.
    """
    calls = [
        _judge_grade(client, _label(e), field, FIELD_RUBRIC, getattr(e, field), getattr(g, field))
        for e, g in pairs
        for field in ("objective", "key_message", "target_audience", "usage_rights")
    ]
    if case.expected.briefs:
        produced = result.intakes[-1].ambiguities if result.intakes else []
        calls.append(
            _judge_grade(
                client,
                None,
                "ambiguities",
                AMBIGUITIES_RUBRIC,
                _bullets(case.expected.ambiguities),
                _bullets(produced),
            )
        )
    async with asyncio.TaskGroup() as group:  # one call raising cancels the rest
        tasks = [group.create_task(call) for call in calls]
    return [task.result() for task in tasks]


async def grade_case(
    case: EvalCase, result: CoordinatorResult, judge_client: AsyncAnthropic
) -> CaseGrades:
    """Every Grade for one Run's result against its Eval Case."""
    pairs = _brief_pairs(case, result)
    grades = _outcome_grades(case, result) + _field_grades(pairs)
    grades += await _judge_grades(case, result, judge_client, pairs)
    return CaseGrades(case_id=case.id, grades=grades)
