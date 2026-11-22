import json

import pytest


async def test_get_asset_known_id_returns_asset_as_structured_content(connect):
    async with connect() as client:
        result = await client.call_tool("dam_get_asset", {"asset_id": "AST-001"})

    assert not result.is_error
    assert result.structured_content == {
        "id": "AST-001",
        "brand": "loomhouse",
        "campaign_name": "Spring Linen Edit",
        "asset_type": "image",
        "channel": "instagram",
        "format_or_dimensions": "1080x1350",
        "file_name": "loomhouse_spring-linen_ig-feed_01.jpg",
        "status": "final",
        "usage_rights": {
            "territories": ["UK", "EU"],
            "channels": ["instagram", "facebook"],
            "expires_on": "2031-03-31",
        },
        "tags": ["linen", "womenswear", "outdoor"],
    }


async def test_get_asset_listed_tool_has_output_schema(connect):
    async with connect() as client:
        tools = await client.list_tools()

    [tool] = [t for t in tools.tools if t.name == "dam_get_asset"]
    assert tool.output_schema is not None
    assert "usage_rights" in tool.output_schema["properties"]


async def test_get_asset_unknown_id_returns_not_found_tool_error(connect):
    async with connect() as client:
        result = await client.call_tool("dam_get_asset", {"asset_id": "AST-999"})

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "not_found"
    assert error["retryable"] is False
    assert "AST-999" in error["message"]
    assert error["details"] == {"asset_id": "AST-999"}


@pytest.mark.parametrize("arguments", [{}, {"asset_id": 5}])
async def test_get_asset_bad_arguments_returns_invalid_input_tool_error(connect, arguments):
    async with connect() as client:
        result = await client.call_tool("dam_get_asset", arguments)

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "invalid_input"
    assert error["retryable"] is False
    assert "asset_id" in error["message"]
    assert [e["field"] for e in error["details"]["errors"]] == ["asset_id"]


async def _search(client, **filters):
    result = await client.call_tool("dam_search_assets", filters)
    assert not result.is_error
    return result.structured_content["assets"]


async def test_search_no_filters_returns_every_asset(connect):
    async with connect() as client:
        assets = await _search(client)

    assert {a["id"] for a in assets} == {"AST-001", "AST-002", "AST-003"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brand", "loomhouse"),
        ("campaign_name", "City Miles"),
        ("asset_type", "image"),
        ("channel", "website"),
        ("status", "final"),
    ],
)
async def test_search_single_filter_returns_only_matching_assets(connect, field, value):
    async with connect() as client:
        assets = await _search(client, **{field: value})

    assert assets
    assert all(a[field] == value for a in assets)


async def test_search_single_filter_narrows_results(connect):
    async with connect() as client:
        everything = await _search(client)
        images = await _search(client, asset_type="image")

    assert 0 < len(images) < len(everything)


async def test_search_filters_combine_with_and(connect):
    async with connect() as client:
        both = await _search(client, asset_type="image", channel="instagram")
        clash = await _search(client, brand="loomhouse", channel="tiktok")

    assert [a["id"] for a in both] == ["AST-001"]
    assert clash == []


async def test_search_matching_ignores_case(connect):
    async with connect() as client:
        lower = await _search(client, brand="loomhouse", channel="instagram")
        mixed = await _search(client, brand="LoomHouse", channel="Instagram")

    assert lower
    assert mixed == lower


async def test_search_no_match_returns_empty_list(connect):
    async with connect() as client:
        assets = await _search(client, status="draft", campaign_name="No Such Campaign")

    assert assets == []


async def test_search_unknown_brand_returns_not_found_tool_error(connect):
    async with connect() as client:
        result = await client.call_tool("dam_search_assets", {"brand": "loomhous"})

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "not_found"
    assert error["retryable"] is False
    assert "loomhous" in error["message"]
    assert error["details"] == {"brand": "loomhous"}


async def test_search_listed_tool_has_output_schema(connect):
    async with connect() as client:
        tools = await client.list_tools()

    [tool] = [t for t in tools.tools if t.name == "dam_search_assets"]
    assert tool.output_schema is not None
    assert "assets" in tool.output_schema["properties"]
