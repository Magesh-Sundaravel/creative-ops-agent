"""Intake agent: a Brief in, one Structured Brief per campaign out."""

from datetime import date

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import StructuredBrief, SubmitBrief
from creative_ops.tools.tool import Tool

SYSTEM_PROMPT = """\
You are an intake agent for a creative agency. Read the Brief and record it by calling \
`submit_brief`, with one Structured Brief per campaign the Brief names. Extract only what \
the Brief states; leave any field it does not give empty. An ambiguity quotes something \
the Brief states that is unclear or contradicts itself. An empty field or an unstated \
detail is not an ambiguity. Today's date is {today}.

Rules:
- Resolve a date with no year against today's date, to its next occurrence on or after today. \
This is the only inference allowed.
- Leave a relative or event-based date (such as "before Black Friday") empty, and add an \
ambiguity quoting the Brief.
- Leave a budget range empty, and add an ambiguity quoting the Brief.
- Keep a past date as stated, and add an ambiguity.
- Never derive the deadline from deliverable due dates; leave it empty if the Brief gives \
only per-deliverable dates.
- If a deliverable's due date is after the deadline, add an ambiguity.
- Record the campaign name without the season, brand or any other qualifier the Brief \
puts beside it: "Resort Shirting (SS29)" is campaign name "Resort Shirting", season "SS29".
- Copy a fact stated once for all campaigns (such as a shared deadline, brand or usage \
rights) into each campaign's Structured Brief.
- If the text is not a Brief, submit no Structured Briefs."""


class NoValidSubmission(RuntimeError):
    """The Run ended without a valid `submit_brief` call.

    Carries the Run's `LoopResult`, so the failed Run can still be inspected.
    """

    def __init__(self, loop_result: LoopResult) -> None:
        super().__init__("Run ended without a valid submit_brief call")
        self.loop_result = loop_result


class CampaignIntake(BaseModel):
    """One campaign's Structured Brief, its Missing Fields and which of them are critical."""

    brief: StructuredBrief
    missing_fields: list[str]
    critical_fields: list[str]
    critical_missing: bool


class IntakeResult(BaseModel):
    """The outcome of intake: one entry per campaign, none if the text is not a Brief."""

    campaigns: list[CampaignIntake]
    ambiguities: list[str]
    loop_result: LoopResult


_CRITICAL_FIELDS = {"brand", "objective", "deadline", "deliverables"}
_CRITICAL_DELIVERABLE_FIELDS = {"asset_type", "channel"}


def _missing(brief: StructuredBrief) -> list[tuple[str, bool]]:
    """Each Missing Field as (path, is_critical), nested gaps as paths like `budget.currency`."""
    missing: list[tuple[str, bool]] = []
    for name, value in brief:
        critical = name in _CRITICAL_FIELDS
        if name == "deliverables":
            if not value:
                missing.append((name, critical))
            for i, item in enumerate(value):
                missing += [
                    (f"deliverables[{i}].{field}", field in _CRITICAL_DELIVERABLE_FIELDS)
                    for field, field_value in item
                    if field_value is None
                ]
        elif name == "budget" and value is not None:
            missing += [(f"budget.{field}", False) for field, v in value if v is None]
        elif value is None or value == []:
            missing.append((name, critical))
    return missing


async def intake(client: AsyncAnthropic, brief: str, today: date | None = None) -> IntakeResult:
    """Turn a Brief (plain text) into one Structured Brief per campaign."""
    today = today or date.today()
    submissions: list[SubmitBrief] = []

    async def handler(args: SubmitBrief) -> str:
        submissions.append(args)
        return "Brief recorded."

    tool = Tool(
        name="submit_brief",
        description=(
            "Submit one Structured Brief per campaign in the Brief, with the Brief's ambiguities."
        ),
        input_model=SubmitBrief,
        handler=handler,
    )
    loop_result = await run(
        client, brief, [tool], system=SYSTEM_PROMPT.format(today=today.isoformat()), agent="intake"
    )
    if not submissions:
        raise NoValidSubmission(loop_result)
    last = submissions[-1]
    return IntakeResult(
        campaigns=[_campaign(brief) for brief in last.briefs],
        ambiguities=last.ambiguities,
        loop_result=loop_result,
    )


def _campaign(brief: StructuredBrief) -> CampaignIntake:
    missing = _missing(brief)
    return CampaignIntake(
        brief=brief,
        missing_fields=[path for path, _ in missing],
        critical_fields=[path for path, critical in missing if critical],
        critical_missing=any(critical for _, critical in missing),
    )
