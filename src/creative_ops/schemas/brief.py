"""Structured Brief: the fields of one campaign in a Brief, holding only what the Brief states."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _blank_to_none(value: Any) -> Any:
    """A blank string means the Brief gave no value."""
    return None if isinstance(value, str) and not value.strip() else value


OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]


class Deliverable(BaseModel):
    """One piece of work the Brief asks for."""

    asset_type: OptionalStr = None
    channel: OptionalStr = Field(
        None,
        description=(
            'Platform name only, such as "instagram". '
            "The placement (feed, story, header) is not a channel."
        ),
    )
    format_or_dimensions: OptionalStr = Field(
        None, description='WxH only, such as "1080x1350", with no placement or label.'
    )
    quantity: int | None = None
    due_date: date | None = None


class Budget(BaseModel):
    """A single stated amount; `currency` is an ISO code."""

    amount: Decimal | None = None
    currency: OptionalStr = None


class StructuredBrief(BaseModel):
    """Every field is optional: a field the Brief doesn't state stays empty."""

    brand: OptionalStr = None
    campaign_name: OptionalStr = None
    season: OptionalStr = None
    objective: OptionalStr = None
    target_audience: OptionalStr = None
    key_message: OptionalStr = None
    usage_rights: OptionalStr = None
    deliverables: list[Deliverable] = []
    mandatories: list[str] = []
    references: list[str] = []
    deadline: date | None = None
    budget: Budget | None = None


class SubmitBrief(BaseModel):
    """Input of the `submit_brief` tool: one Structured Brief per campaign.

    Ambiguities stay at Brief level, outside the Structured Briefs.
    """

    briefs: list[StructuredBrief]
    ambiguities: list[str] = []
