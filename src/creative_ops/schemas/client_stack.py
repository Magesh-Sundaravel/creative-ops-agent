"""Client-stack data as the agents read it over MCP. Independent of the mock server's models."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel


class BrandRef(BaseModel):
    """One entry of `brand://brands`."""

    slug: str
    display_name: str


class PaletteColour(BaseModel):
    """One named colour in a brand's palette."""

    name: str
    hex: str


class Typography(BaseModel):
    """The typefaces a brand uses for headings and body copy."""

    heading: str
    body: str


class BrandGuidelines(BaseModel):
    """A brand's rules for how its creative work must look and sound."""

    palette: list[PaletteColour]
    typography: Typography
    logo_rules: list[str]
    tone_of_voice: str
    dos: list[str]
    donts: list[str]
    mandatory_legal_lines: list[str]
    approved_channels: list[str]
    channel_dimensions: dict[str, list[str]]


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
    status: str
    usage_rights: UsageRights
    tags: list[str]


class AssetList(BaseModel):
    """The result of `dam_search_assets`."""

    assets: list[Asset]


class Task(BaseModel):
    """A unit of agency work in the project tracker, as `tracker_create_task` returns it."""

    id: str
    title: str
    brand: str
    campaign_name: str
    description: str
    due_date: date | None
    status: str


class ToolError(BaseModel):
    """What a client-stack tool, or a Gate in front of one, returns instead of a result."""

    type: Literal["not_found", "invalid_input", "permission_denied", "upstream_unavailable"]
    message: str
    retryable: bool
    details: dict[str, Any]
