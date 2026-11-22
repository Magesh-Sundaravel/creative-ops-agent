"""Data owned by the mock client stack. Independent of the agent code's own schemas."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel


class Brand(BaseModel):
    """A client brand, identified everywhere by its slug."""

    slug: str
    display_name: str


class UsageRights(BaseModel):
    """The territories, channels and time period in which an Asset may be published."""

    territories: list[str]
    channels: list[str]
    expires_on: date


class Asset(BaseModel):
    """A piece of creative work held in the DAM, described by its metadata only."""

    id: str
    brand: str
    campaign_name: str
    asset_type: str
    channel: str
    format_or_dimensions: str
    file_name: str
    status: Literal["draft", "final"]
    usage_rights: UsageRights
    tags: list[str]


class ToolError(BaseModel):
    """What a tool returns instead of a result when it can't do what was asked."""

    type: Literal["not_found", "invalid_input", "permission_denied", "upstream_unavailable"]
    message: str
    retryable: bool
    details: dict[str, Any]
