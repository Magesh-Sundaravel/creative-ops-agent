"""Review agent: a Structured Brief in, Findings and a Review Verdict out (see ADR 0003)."""

from datetime import date
from typing import Annotated, Any, Literal

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from creative_ops.loop.agent_loop import LoopResult, run
from creative_ops.schemas.brief import StructuredBrief
from creative_ops.schemas.client_stack import AssetList, BrandGuidelines, BrandRef
from creative_ops.tools.mcp_adapter import call_tool, read_resource
from creative_ops.tools.tool import Tool

SYSTEM_PROMPT = """\
You are a review agent for a creative agency. Review the campaign's Assets against the \
Structured Brief and the brand's Brand Guidelines, then record your review by calling \
`submit_review`. Today's date is {today}.

Judge:
- each Asset's territories against the brief's Usage Rights
- each Asset's tags against the brand's dos and don'ts
- how well the Assets fit the brief's objective and key message
- which Assets cover which deliverable

Deliverables are numbered by their position in the brief's `deliverables` list, from 0. \
Every deliverable must appear in the coverage. A deliverable with no covering Asset needs \
a blocking Finding whose subject is its index. Mark a Finding blocking only if the Asset \
can't be delivered as it is; otherwise mark it advisory."""


class CannotReview(ValueError):
    """The brief can't be reviewed; a human has to fix it first. No Run was started."""


class NoValidReview(RuntimeError):
    """The Run ended without a valid `submit_review` call.

    Carries the Run's `LoopResult`, so the failed Run can still be inspected.
    """

    def __init__(self, loop_result: LoopResult) -> None:
        super().__init__("Run ended without a valid submit_review call")
        self.loop_result = loop_result


def _index_to_int(value: Any) -> Any:
    """A deliverable index written as a string ("1") means that index."""
    return int(value) if isinstance(value, str) and value.isdigit() else value


class Finding(BaseModel):
    """One problem with an Asset, or a deliverable with no covering Asset."""

    subject: Annotated[str | int, BeforeValidator(_index_to_int)] = Field(
        description='The Asset id (e.g. "AST-009"), or the deliverable index (e.g. 0).'
    )
    rule: str = Field(description='Short name of the rule broken, e.g. "territory".')
    explanation: str = Field(description="What is wrong and what the agency should fix.")
    severity: Literal["blocking", "advisory"]


class SubmitReview(BaseModel):
    """Input of the `submit_review` tool."""

    findings: list[Finding]
    coverage: dict[int, list[str]] = Field(
        description="For every deliverable index, the ids of the Assets that cover it "
        "(an empty list if none do)."
    )


class ReviewResult(BaseModel):
    """The outcome of a review."""

    brand: str
    findings: list[Finding]
    coverage: dict[int, list[str]]
    verdict: Literal["approved", "revisions_requested"]
    loop_result: LoopResult


async def _resolve_brand(session: Client, brand: str) -> str:
    """The slug of the brand whose slug or display name matches `brand`, ignoring case."""
    brands = await read_resource(session, "brand://brands", TypeAdapter(list[BrandRef]))
    for ref in brands:
        if brand.casefold() in (ref.slug.casefold(), ref.display_name.casefold()):
            return ref.slug
    raise CannotReview(f"Brand {brand!r} is not a known client brand")


def _check_submission(args: SubmitReview, deliverables: int, asset_ids: set[str]) -> None:
    """Raise unless the submission names only Assets under review and deliverables in the
    brief, and the coverage accounts for every deliverable."""
    expected = set(range(deliverables))
    unknown_subjects = [f.subject for f in args.findings if f.subject not in asset_ids | expected]
    if unknown_subjects:
        raise ValueError(
            f"Finding subjects {unknown_subjects} are neither Assets under review "
            f"{sorted(asset_ids)} nor deliverable indices {sorted(expected)}"
        )
    if set(args.coverage) != expected:
        raise ValueError(
            f"coverage must have exactly the deliverable indices {sorted(expected)}, "
            f"got {sorted(args.coverage)}"
        )
    unknown = {i for ids in args.coverage.values() for i in ids} - asset_ids
    if unknown:
        raise ValueError(
            f"coverage names Assets not under review: {sorted(unknown)}. "
            f"Use only {sorted(asset_ids)}"
        )
    flagged = {f.subject for f in args.findings if f.severity == "blocking"}
    uncovered = [i for i, ids in args.coverage.items() if not ids and i not in flagged]
    if uncovered:
        raise ValueError(
            f"deliverables {uncovered} have no covering Asset; add a blocking Finding "
            "for each, with the deliverable index as its subject"
        )


async def review(
    client: AsyncAnthropic, session: Client, brief: StructuredBrief, today: date | None = None
) -> ReviewResult:
    """Review the campaign's Assets against a Structured Brief and the Brand Guidelines."""
    if not brief.brand or not brief.campaign_name:
        raise CannotReview("The brief needs a brand and a campaign name to be reviewed")
    today = today or date.today()
    brand = await _resolve_brand(session, brief.brand)
    found = await call_tool(
        session,
        "dam_search_assets",
        {"brand": brand, "campaign_name": brief.campaign_name},
        AssetList,
    )
    if not found.assets:
        raise CannotReview(f"No Assets for brand {brand!r}, campaign {brief.campaign_name!r}")
    guidelines = await read_resource(
        session, f"brand://guidelines/{brand}", TypeAdapter(BrandGuidelines)
    )
    asset_ids = {asset.id for asset in found.assets}
    submissions: list[SubmitReview] = []

    async def handler(args: SubmitReview) -> str:
        _check_submission(args, len(brief.deliverables), asset_ids)
        submissions.append(args)
        return "Review recorded."

    tool = Tool(
        name="submit_review",
        description=(
            "Record your review: the Findings, and which Assets cover each deliverable. "
            "Call it once you have judged every Asset; call it again only to replace an "
            "earlier submission. It is your only tool. subject: an Asset id or a deliverable "
            "index. severity: blocking if the Asset can't be delivered as it is, else advisory. "
            "coverage: every deliverable index, mapped to the ids of its covering Assets."
        ),
        input_model=SubmitReview,
        handler=handler,
    )
    prompt = (
        f"<structured_brief>\n{brief.model_dump_json(indent=2)}\n</structured_brief>\n"
        f"<brand_guidelines>\n{guidelines.model_dump_json(indent=2)}\n</brand_guidelines>\n"
        f"<assets>\n{found.model_dump_json(indent=2)}\n</assets>"
    )
    loop_result = await run(
        client, prompt, [tool], system=SYSTEM_PROMPT.format(today=today.isoformat())
    )
    if not submissions:
        raise NoValidReview(loop_result)
    last = submissions[-1]
    blocking = any(f.severity == "blocking" for f in last.findings)
    return ReviewResult(
        brand=brand,
        findings=last.findings,
        coverage=last.coverage,
        verdict="revisions_requested" if blocking else "approved",
        loop_result=loop_result,
    )
