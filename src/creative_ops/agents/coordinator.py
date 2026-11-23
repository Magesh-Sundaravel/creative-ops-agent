"""Coordinator: a Brief in, taken through intake, review and handoff (see ADR 0005)."""

from datetime import date
from typing import Literal

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel

from creative_ops.agents.handoff import HandoffPackage, handoff
from creative_ops.agents.intake import IntakeResult, intake
from creative_ops.agents.review import ReviewResult, review
from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import StructuredBrief
from creative_ops.tools.tool import Tool, ToolResultError

SYSTEM_PROMPT = """\
You are the coordinator for a creative agency. Take the Brief from intake to handoff with \
your tools: call `intake` once to turn the Brief into one Structured Brief per campaign, \
then `review` each campaign, then `handoff` each campaign whose Review Verdict is approved. \
Name a campaign by its brand and campaign name exactly as `intake` returns them. When there \
is nothing left to do, stop."""

MAX_ITERATIONS = 8


class NoInput(BaseModel):
    """Input of a tool that takes none."""


class CampaignKey(BaseModel):
    """Names one campaign, as `intake` returned it."""

    brand: str
    campaign_name: str


class CampaignSummary(CampaignKey):
    """What the model sees of one campaign after intake."""

    missing_fields: list[str]


class IntakeSummary(BaseModel):
    """What the model sees of an intake."""

    campaigns: list[CampaignSummary]
    ambiguities: list[str]


class ReviewSummary(BaseModel):
    """What the model sees of a review."""

    verdict: Literal["approved", "revisions_requested"]
    blocking_findings: int


class HandoffSummary(BaseModel):
    """What the model sees of a handoff."""

    task_id: str


class CoordinatorResult(BaseModel):
    """Built by code from what the tools recorded, never from the model's final text.

    Every subagent `LoopResult` is in the recorded intakes, reviews and handoffs.
    """

    handoffs: list[HandoffPackage]
    intakes: list[IntakeResult]
    reviews: list[ReviewResult]
    loop_result: LoopResult


async def coordinate(
    client: AsyncAnthropic, session: Client, brief: str, today: date | None = None
) -> CoordinatorResult:
    """Take a Brief through intake, review and handoff."""
    intakes: list[IntakeResult] = []
    reviews: list[ReviewResult] = []
    # The current review of each campaign; a new intake makes every earlier review stale.
    current_review: dict[tuple[str, str], ReviewResult] = {}
    handoffs: list[HandoffPackage] = []
    briefs: dict[tuple[str | None, str | None], StructuredBrief] = {}

    def campaign(key: CampaignKey) -> StructuredBrief:
        found = briefs.get((key.brand, key.campaign_name))
        if found is None:
            raise ToolResultError(
                f"No campaign {key.brand!r} / {key.campaign_name!r}. "
                f"Use a brand and campaign name returned by intake: {sorted(map(str, briefs))}"
            )
        return found

    async def run_intake(_: NoInput) -> str:
        result = await intake(client, brief, today)
        intakes.append(result)
        briefs.clear()
        current_review.clear()
        briefs.update({(c.brief.brand, c.brief.campaign_name): c.brief for c in result.campaigns})
        return IntakeSummary(
            campaigns=[
                CampaignSummary(
                    brand=str(c.brief.brand),
                    campaign_name=str(c.brief.campaign_name),
                    missing_fields=c.missing_fields,
                )
                for c in result.campaigns
            ],
            ambiguities=result.ambiguities,
        ).model_dump_json()

    async def run_review(key: CampaignKey) -> str:
        result = await review(client, session, campaign(key), today)
        reviews.append(result)
        current_review[(key.brand, key.campaign_name)] = result
        return ReviewSummary(
            verdict=result.verdict,
            blocking_findings=sum(f.severity == "blocking" for f in result.findings),
        ).model_dump_json()

    async def run_handoff(key: CampaignKey) -> str:
        structured = campaign(key)
        reviewed = current_review.get((key.brand, key.campaign_name))
        if reviewed is None:
            raise ToolResultError("Review this campaign since its latest intake first.")
        package = await handoff(client, session, structured, reviewed)
        handoffs.append(package)
        return HandoffSummary(task_id=package.task_id).model_dump_json()

    tools = [
        Tool(
            name="intake",
            description="Turn the Brief into one Structured Brief per campaign.",
            input_model=NoInput,
            handler=run_intake,
        ),
        Tool(
            name="review",
            description="Review one campaign's Assets. Returns its Review Verdict.",
            input_model=CampaignKey,
            handler=run_review,
        ),
        Tool(
            name="handoff",
            description="Hand one approved campaign to production. Returns the created Task.",
            input_model=CampaignKey,
            handler=run_handoff,
        ),
    ]
    loop_result = await run(
        client,
        f"<brief>\n{brief}\n</brief>",
        tools,
        system=SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
    )
    return CoordinatorResult(
        handoffs=handoffs,
        intakes=intakes,
        reviews=reviews,
        loop_result=loop_result,
    )
