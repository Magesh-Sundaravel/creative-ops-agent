import json


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
