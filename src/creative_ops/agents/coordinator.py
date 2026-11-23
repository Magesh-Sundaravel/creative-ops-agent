"""Coordinator: a Brief in, taken through intake, review and handoff (see ADR 0005)."""

from datetime import date
from typing import Literal

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel

from creative_ops.agents.handoff import HandoffPackage, handoff
from creative_ops.agents.intake import IntakeResult, intake
from creative_ops.agents.review import CannotReview, ReviewResult, review
from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import StructuredBrief
from creative_ops.schemas.client_stack import permission_denied
from creative_ops.tools.tool import Tool, ToolResultError

SYSTEM_PROMPT = """\
You are the coordinator for a creative agency. Take the Brief from intake to handoff with \
your tools: call `intake` once to turn the Brief into one Structured Brief per campaign, \
then `review` each campaign, then `handoff` each campaign whose Review Verdict is approved. \
Name a campaign by its brand and campaign name exactly as `intake` returns them. A campaign \
marked `escalated` is waiting for a human: don't review or hand it off. When there is nothing \
left to do, stop."""

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
    escalated: bool


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


EscalationReason = Literal["missing_critical_field", "cannot_review", "not_a_brief"]


class Escalation(BaseModel):
    """A decision handed to a human: why, the kind of answer it takes, and what it needs."""

    reason: EscalationReason
    answer: Literal["amend"]
    needs: list[str]
    ambiguities: list[str]


class CampaignOutcome(BaseModel):
    """What became of one campaign intake returned; neither outcome means it is unfinished."""

    brand: str | None
    campaign_name: str | None
    handoff: HandoffPackage | None = None
    escalation: Escalation | None = None


class CoordinatorState(BaseModel):
    """Everything `resume()` needs to continue after an Escalation; serialisable."""

    brief: str
    escalations: list[Escalation]


class CoordinatorResult(BaseModel):
    """Built by code from what the tools recorded, never from the model's final text.

    Every subagent `LoopResult` is in the recorded intakes, reviews and Handoff Packages.
    """

    campaigns: list[CampaignOutcome]
    escalation: Escalation | None
    """A Brief-level Escalation: no campaign could start."""
    intakes: list[IntakeResult]
    reviews: list[ReviewResult]
    loop_result: LoopResult
    state: CoordinatorState | None
    """Set when there is any Escalation, to pass to `resume()`."""


async def coordinate(
    client: AsyncAnthropic, session: Client, brief: str, today: date | None = None
) -> CoordinatorResult:
    """Take a Brief through intake, review and handoff."""
    intakes: list[IntakeResult] = []
    reviews: list[ReviewResult] = []
    current_review: dict[tuple[str, str], ReviewResult] = {}
    intake_claimed = False
    briefs: dict[tuple[str | None, str | None], StructuredBrief] = {}
    outcomes: dict[tuple[str | None, str | None], CampaignOutcome] = {}
    brief_escalation: Escalation | None = None

    def campaign(key: CampaignKey) -> StructuredBrief:
        """The Structured Brief intake returned for this campaign."""
        found = briefs.get((key.brand, key.campaign_name))
        if found is None:
            raise ToolResultError(
                f"No campaign {key.brand!r} / {key.campaign_name!r}. "
                f"Use a brand and campaign name returned by intake: {sorted(map(str, briefs))}"
            )
        return found

    def escalation(reason: EscalationReason, needs: list[str]) -> Escalation:
        """An Escalation carrying the Brief's Ambiguities."""
        return Escalation(
            reason=reason, answer="amend", needs=needs, ambiguities=intakes[-1].ambiguities
        )

    async def run_intake(_: NoInput) -> str:
        """Run Intake, once per Coordinator Run, so briefs never change under review or handoff."""
        nonlocal intake_claimed, brief_escalation
        # Claim intake before the first await, so two calls in one response can't both run.
        if intake_claimed:
            raise ToolResultError(
                "Intake has already run, or is running, for this Brief. Use its campaigns; "
                "call intake again only if it failed."
            )
        intake_claimed = True
        succeeded = False
        try:
            result = await intake(client, brief, today)
            succeeded = True
        finally:
            # A failed Run recorded no campaigns, so intake may run again.
            intake_claimed = succeeded
        intakes.append(result)
        for c in result.campaigns:
            key = (c.brief.brand, c.brief.campaign_name)
            briefs[key] = c.brief
            outcomes[key] = CampaignOutcome(brand=key[0], campaign_name=key[1])
            if c.critical_missing:
                outcomes[key].escalation = escalation("missing_critical_field", c.critical_fields)
        if not result.campaigns:
            brief_escalation = escalation("not_a_brief", ["A Brief naming at least one campaign"])
        return IntakeSummary(
            campaigns=[
                CampaignSummary(
                    brand=str(c.brief.brand),
                    campaign_name=str(c.brief.campaign_name),
                    missing_fields=c.missing_fields,
                    escalated=c.critical_missing,
                )
                for c in result.campaigns
            ],
            ambiguities=result.ambiguities,
        ).model_dump_json()

    async def run_review(key: CampaignKey) -> str:
        """Run Review on one campaign and record its result, unless it is escalated (Gate)."""
        structured = campaign(key)
        escalated = outcomes[(key.brand, key.campaign_name)].escalation
        if escalated is not None:
            raise ToolResultError(
                permission_denied(f"This campaign is escalated to a human ({escalated.reason}).")
            )
        try:
            result = await review(client, session, structured, today)
        except CannotReview as exc:
            outcomes[(key.brand, key.campaign_name)].escalation = escalation(
                "cannot_review", [str(exc)]
            )
            raise ToolResultError(permission_denied(f"Escalated to a human: {exc}")) from exc
        reviews.append(result)
        current_review[(key.brand, key.campaign_name)] = result
        return ReviewSummary(
            verdict=result.verdict,
            blocking_findings=sum(f.severity == "blocking" for f in result.findings),
        ).model_dump_json()

    async def run_handoff(key: CampaignKey) -> str:
        """Run Handoff on one reviewed campaign and record its Handoff Package."""
        structured = campaign(key)
        reviewed = current_review.get((key.brand, key.campaign_name))
        if reviewed is None:
            raise ToolResultError("Review this campaign before handing it off.")
        package = await handoff(client, session, structured, reviewed)
        outcomes[(key.brand, key.campaign_name)].handoff = package
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
    campaigns = list(outcomes.values())
    escalations = [
        e for e in [brief_escalation, *(c.escalation for c in campaigns)] if e is not None
    ]
    return CoordinatorResult(
        campaigns=campaigns,
        escalation=brief_escalation,
        intakes=intakes,
        reviews=reviews,
        loop_result=loop_result,
        state=CoordinatorState(brief=brief, escalations=escalations) if escalations else None,
    )


class Amend(BaseModel):
    """A human's answer to an Escalation: text appended to the Brief, which Intake re-runs on."""

    kind: Literal["amend"] = "amend"
    text: str


async def resume(
    client: AsyncAnthropic,
    session: Client,
    state: CoordinatorState,
    answer: Amend,
    today: date | None = None,
) -> CoordinatorResult:
    """Continue after an Escalation: a new Coordinator Run on the amended Brief."""
    return await coordinate(client, session, f"{state.brief}\n\n{answer.text}", today)
