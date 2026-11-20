"""Structured Brief: the fields of a Brief, holding only what the Brief states."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator


def _blank_to_none(value: Any) -> Any:
    """A blank string means the Brief gave no value."""
    return None if isinstance(value, str) and not value.strip() else value


OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]


class Deliverable(BaseModel):
    """One piece of work the Brief asks for."""

    asset_type: OptionalStr = None
    channel: OptionalStr = None
    format_or_dimensions: OptionalStr = None
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
    """Input of the `submit_brief` tool; keeps Ambiguities out of the Structured Brief."""

    brief: StructuredBrief
    ambiguities: list[str] = []
