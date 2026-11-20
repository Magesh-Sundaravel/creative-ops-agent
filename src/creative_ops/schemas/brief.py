"""Structured Brief: the fields of a Brief, holding only what the Brief states."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class Deliverable(BaseModel):
    """One piece of work the Brief asks for."""

    asset_type: str | None = None
    channel: str | None = None
    format_or_dimensions: str | None = None
    quantity: int | None = None
    due_date: date | None = None


class Budget(BaseModel):
    """A single stated amount; `currency` is an ISO code."""

    amount: Decimal | None = None
    currency: str | None = None


class StructuredBrief(BaseModel):
    """Every field is optional: a field the Brief doesn't state stays empty."""

    brand: str | None = None
    campaign_name: str | None = None
    season: str | None = None
    objective: str | None = None
    target_audience: str | None = None
    key_message: str | None = None
    usage_rights: str | None = None
    deliverables: list[Deliverable] = []
    mandatories: list[str] = []
    references: list[str] = []
    deadline: date | None = None
    budget: Budget | None = None


class SubmitBrief(BaseModel):
    """Input of the `submit_brief` tool; keeps Ambiguities out of the Structured Brief."""

    brief: StructuredBrief
    ambiguities: list[str] = []
