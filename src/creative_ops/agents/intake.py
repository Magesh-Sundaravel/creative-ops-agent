"""Intake agent: a Brief in, a Structured Brief out."""

from datetime import date

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import StructuredBrief, SubmitBrief
from creative_ops.tools.tool import Tool

SYSTEM_PROMPT = """\
You are an intake agent for a creative agency. Read the Brief and record it by calling \
`submit_brief`. Extract only what the Brief states; leave any field it does not give empty. \
List anything unclear or contradictory as an ambiguity. Today's date is {today}."""


class NoValidSubmission(RuntimeError):
    """The Run ended without a valid `submit_brief` call.

    Carries the Run's `LoopResult`, so the failed Run can still be inspected.
    """

    def __init__(self, loop_result: LoopResult) -> None:
        super().__init__("Run ended without a valid submit_brief call")
        self.loop_result = loop_result


class IntakeResult(BaseModel):
    """The outcome of intake."""

    brief: StructuredBrief
    missing_fields: list[str]
    ambiguities: list[str]
    loop_result: LoopResult


def _missing_fields(brief: StructuredBrief) -> list[str]:
    """Top-level fields the Brief gives no value for (empty or an empty list)."""
    return [name for name, value in brief if value is None or value == []]


async def intake(client: AsyncAnthropic, brief: str, today: date | None = None) -> IntakeResult:
    """Turn a Brief (plain text) into a Structured Brief."""
    today = today or date.today()
    submissions: list[SubmitBrief] = []

    async def handler(args: SubmitBrief) -> str:
        submissions.append(args)
        return "Brief recorded."

    tool = Tool(
        name="submit_brief",
        description="Submit the Structured Brief extracted from the Brief, with any ambiguities.",
        input_model=SubmitBrief,
        handler=handler,
    )
    loop_result = await run(
        client, brief, [tool], system=SYSTEM_PROMPT.format(today=today.isoformat())
    )
    if not submissions:
        raise NoValidSubmission(loop_result)
    last = submissions[-1]
    return IntakeResult(
        brief=last.brief,
        missing_fields=_missing_fields(last.brief),
        ambiguities=last.ambiguities,
        loop_result=loop_result,
    )
