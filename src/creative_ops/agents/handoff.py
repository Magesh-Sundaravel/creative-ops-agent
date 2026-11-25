"""Handoff agent: an approved review in, a Handoff Package and a tracker Task out (ADR 0004)."""

from datetime import date

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel, Field, TypeAdapter

from creative_ops.agents.review import Finding, ReviewResult
from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import Deliverable, StructuredBrief
from creative_ops.schemas.client_stack import (
    Asset,
    AssetList,
    BrandGuidelines,
    Task,
    ToolError,
    permission_denied,
)
from creative_ops.tools.mcp_adapter import ClientStackError, call_tool, mcp_tools, read_resource
from creative_ops.tools.tool import Tool, ToolResultError

SYSTEM_PROMPT = """\
You are a handoff agent for a creative agency. The campaign's Assets have passed review. \
Hand the work to production: create one Task in the tracker with `tracker_create_task`, \
then record your production notes with `submit_handoff`.

The <handoff_facts> are already settled: which Assets cover each deliverable, the mandatory \
legal lines, the Usage Rights every covering Asset shares, the deadline and advisory notes. \
Write the Task's title and description and your production notes from them."""


class CannotHandoff(ValueError):
    """The review isn't approved or covers no Assets, so there is nothing to hand off.

    No Run was started.
    """


class HandoffIncomplete(RuntimeError):
    """The Run ended without a created Task or without a valid `submit_handoff` call.

    Carries the Run's `LoopResult`, so the failed Run can still be inspected.
    """

    def __init__(self, loop_result: LoopResult) -> None:
        super().__init__("Run ended without a created Task and a valid submit_handoff call")
        self.loop_result = loop_result


class CreateTask(BaseModel):
    """Model-facing input of the gated `tracker_create_task`; code fills in the rest."""

    title: str = Field(description='Short name for the work, e.g. "Spring Linen Edit: social".')
    description: str = Field(description="What production has to make.")


class SubmitHandoff(BaseModel):
    """Input of the `submit_handoff` tool."""

    production_notes: str = Field(description="Context for production beyond the facts given.")


class AssetRef(BaseModel):
    """An Asset as production needs to find it."""

    id: str
    file_name: str


class PackagedDeliverable(BaseModel):
    """One deliverable in the brief and the Assets that cover it."""

    deliverable: Deliverable
    assets: list[AssetRef]


class RightsSummary(BaseModel):
    """The Usage Rights shared by every covering Asset, and the earliest expiry."""

    territories: list[str]
    channels: list[str]
    expires_on: date


class HandoffFacts(BaseModel):
    """The parts of a Handoff Package that code assembles from verified data."""

    brand: str
    campaign_name: str
    deliverables: list[PackagedDeliverable]
    legal_lines: list[str]
    usage_rights: RightsSummary
    deadline: date | None
    notes: list[Finding]


class HandoffPackage(HandoffFacts):
    """Everything production needs to deliver an approved campaign."""

    production_notes: str
    task_id: str
    loop_result: LoopResult


def _shared(lists: list[list[str]]) -> list[str]:
    """The items in every list, in the order of the first."""
    first, *rest = lists
    return [item for item in first if all(item in other for other in rest)]


async def _assemble_facts(
    session: Client, brief: StructuredBrief, review: ReviewResult
) -> HandoffFacts:
    """The package facts, read from the client stack rather than written by the model."""
    found = await call_tool(
        session,
        "dam_search_assets",
        {"brand": review.brand, "campaign_name": brief.campaign_name},
        AssetList,
    )
    assets = {asset.id: asset for asset in found.assets}
    guidelines = await read_resource(
        session, f"brand://guidelines/{review.brand}", TypeAdapter(BrandGuidelines)
    )
    covering: list[Asset] = [
        assets[asset_id] for i in sorted(review.coverage) for asset_id in review.coverage[i]
    ]
    return HandoffFacts(
        brand=review.brand,
        campaign_name=brief.campaign_name,
        deliverables=[
            PackagedDeliverable(
                deliverable=deliverable,
                assets=[
                    AssetRef(id=asset_id, file_name=assets[asset_id].file_name)
                    for asset_id in review.coverage[i]
                ],
            )
            for i, deliverable in enumerate(brief.deliverables)
        ],
        legal_lines=guidelines.mandatory_legal_lines,
        usage_rights=RightsSummary(
            territories=_shared([a.usage_rights.territories for a in covering]),
            channels=_shared([a.usage_rights.channels for a in covering]),
            expires_on=min(a.usage_rights.expires_on for a in covering),
        ),
        deadline=brief.deadline,
        notes=[f for f in review.findings if f.severity == "advisory"],
    )


async def handoff(
    client: AsyncAnthropic, session: Client, brief: StructuredBrief, review: ReviewResult
) -> HandoffPackage:
    """Package an approved campaign for production and create its tracker Task."""
    if review.verdict != "approved":
        raise CannotHandoff(f"Review Verdict is {review.verdict!r}, not 'approved'")
    if not any(review.coverage.values()):
        raise CannotHandoff("The review covers no Assets")
    facts = await _assemble_facts(session, brief, review)
    created: list[Task] = []
    submissions: list[SubmitHandoff] = []
    slot_taken = False

    async def create_task(args: CreateTask) -> str:
        """The Gate on `tracker_create_task`: one successful create per Run (ADR 0004)."""
        nonlocal slot_taken
        if review.verdict != "approved":
            raise ToolResultError(
                permission_denied(
                    f"Review Verdict is {review.verdict!r}; only approved work is handed off."
                )
            )
        # Take the slot before the first await, so concurrent calls in one response can't race.
        if slot_taken:
            raise ToolResultError(
                permission_denied(
                    "This Run has already used its one tracker_create_task call, or that call's "
                    "outcome is unknown. Do not create another Task; call submit_handoff."
                )
            )
        slot_taken = True
        try:
            task = await call_tool(
                session,
                "tracker_create_task",
                {
                    **args.model_dump(),
                    "brand": review.brand,
                    "campaign_name": brief.campaign_name,
                    "due_date": brief.deadline.isoformat() if brief.deadline else None,
                },
                Task,
            )
        except ClientStackError as exc:
            # Free the slot only when the error proves nothing was written; after
            # upstream_unavailable the Task may exist, so a retry could duplicate it.
            if ToolError.model_validate_json(exc.error).type != "upstream_unavailable":
                slot_taken = False
            raise ToolResultError(exc.error) from exc
        created.append(task)
        return task.model_dump_json()

    async def submit(args: SubmitHandoff) -> str:
        submissions.append(args)
        return "Handoff recorded."

    tools = [
        *await mcp_tools(session, ["dam_get_asset"]),
        Tool(
            name="tracker_create_task",
            description=(
                "Create the campaign's Task in the project tracker. Brand, campaign and due "
                "date are filled in for you. Returns the created Task."
            ),
            input_model=CreateTask,
            handler=create_task,
        ),
        Tool(
            name="submit_handoff",
            description="Record your production notes. Call it after creating the Task.",
            input_model=SubmitHandoff,
            handler=submit,
        ),
    ]
    prompt = (
        f"<structured_brief>\n{brief.model_dump_json(indent=2)}\n</structured_brief>\n"
        f"<handoff_facts>\n{facts.model_dump_json(indent=2)}\n</handoff_facts>"
    )
    loop_result = await run(client, prompt, tools, system=SYSTEM_PROMPT, agent="handoff")
    if not created or not submissions:
        raise HandoffIncomplete(loop_result)
    return HandoffPackage(
        **dict(facts),
        production_notes=submissions[-1].production_notes,
        task_id=created[-1].id,
        loop_result=loop_result,
    )
