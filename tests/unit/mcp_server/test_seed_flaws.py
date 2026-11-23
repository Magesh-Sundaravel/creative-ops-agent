"""The seed's deliberate flaws, found through the MCP session, so Review has something to catch."""

import json


async def _search(client, **filters):
    result = await client.call_tool("dam_search_assets", filters)
    assert not result.is_error
    return result.structured_content["assets"]


async def _get(client, asset_id):
    result = await client.call_tool("dam_get_asset", {"asset_id": asset_id})
    assert not result.is_error
    return result.structured_content


def _is_landscape(dimensions):
    width, height = (int(n) for n in dimensions.split("x"))
    return width > height


async def test_seed_has_about_twenty_assets_across_every_brand(connect):
    async with connect() as client:
        assets = await _search(client)

    assert len(assets) == 20
    assert {a["brand"] for a in assets} == {"loomhouse", "stridewell", "verdant-skin"}
    assert 2 <= len({a["campaign_name"] for a in assets}) <= 3


async def test_seed_flaw_wrong_dimensions_landscape_video_on_tiktok(connect):
    async with connect() as client:
        tiktok = await _search(client, channel="tiktok")

    assert [a["id"] for a in tiktok if _is_landscape(a["format_or_dimensions"])] == ["AST-014"]


async def test_seed_flaw_expired_usage_rights_uses_fixed_past_date(connect):
    async with connect() as client:
        assets = await _search(client)

    expiring_before_2030 = {
        a["id"]: a["usage_rights"]["expires_on"]
        for a in assets
        if a["usage_rights"]["expires_on"] < "2030-01-01"
    }
    assert expiring_before_2030 == {"AST-020": "2025-06-30"}


async def test_seed_flaw_wrong_territory_rights_share_none_with_rest_of_campaign(connect):
    async with connect() as client:
        flawed = await _get(client, "AST-015")
        campaign = await _search(client, campaign_name=flawed["campaign_name"])

    campaign_territories = {
        t for a in campaign if a["id"] != "AST-015" for t in a["usage_rights"]["territories"]
    }
    assert campaign_territories
    assert campaign_territories.isdisjoint(flawed["usage_rights"]["territories"])


async def test_seed_flaw_draft_among_finals_in_same_campaign(connect):
    async with connect() as client:
        drafts = await _search(client, status="draft")
        campaign = await _search(client, campaign_name=drafts[0]["campaign_name"])

    assert [a["id"] for a in drafts] == ["AST-009"]
    assert "final" in {a["status"] for a in campaign}


async def test_seed_flaw_channel_not_in_brands_approved_channels(connect):
    async with connect() as client:
        flawed = await _get(client, "AST-019")
        resource = await client.read_resource(f"brand://guidelines/{flawed['brand']}")

    approved = json.loads(resource.contents[0].text)["approved_channels"]
    assert flawed["channel"] not in approved
