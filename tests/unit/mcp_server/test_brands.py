import json

import pytest
from mcp import MCPError

from mcp_server.models import BrandGuidelines


async def test_brands_resource_lists_slug_and_display_name_for_every_brand(connect):
    async with connect() as client:
        result = await client.read_resource("brand://brands")

    [content] = result.contents
    assert content.mime_type == "application/json"
    assert json.loads(content.text) == [
        {"slug": "loomhouse", "display_name": "Loomhouse"},
        {"slug": "stridewell", "display_name": "Stridewell"},
        {"slug": "verdant-skin", "display_name": "Verdant Skin"},
    ]


@pytest.mark.parametrize(
    ("slug", "heading_font"),
    [("loomhouse", "Canela"), ("stridewell", "Druk Wide"), ("verdant-skin", "Recoleta")],
)
async def test_guidelines_resource_known_brand_returns_its_brand_guidelines_json(
    connect, slug, heading_font
):
    async with connect() as client:
        result = await client.read_resource(f"brand://guidelines/{slug}")

    [content] = result.contents
    assert content.mime_type == "application/json"
    guidelines = BrandGuidelines.model_validate_json(content.text)
    assert guidelines.typography.heading == heading_font
    assert guidelines.palette
    assert guidelines.approved_channels
    assert guidelines.channel_dimensions.keys() == set(guidelines.approved_channels)


async def test_guidelines_resource_is_listed_as_template(connect):
    async with connect() as client:
        templates = await client.list_resource_templates()

    assert [t.uri_template for t in templates.resource_templates] == ["brand://guidelines/{brand}"]


async def test_guidelines_resource_unknown_brand_fails_naming_slug(connect):
    async with connect() as client:
        with pytest.raises(MCPError, match="loomhous") as exc_info:
            await client.read_resource("brand://guidelines/loomhous")

    assert "loomhouse" in str(exc_info.value)  # lists the known brands
