"""The seed's deliberate flaws, found through the MCP session, so Review has something to catch."""

import json


async def _search(client, **filters):
    result = await client.call_tool("dam_search_assets", filters)
    assert not result.is_error
    return result.structured_content["assets"]


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
        assets = await _search(client)

    def rest_of_campaign_territories(asset):
        return {
            t
            for a in assets
            if a["campaign_name"] == asset["campaign_name"] and a["id"] != asset["id"]
            for t in a["usage_rights"]["territories"]
        }

    clashing = [
        a["id"]
        for a in assets
        if rest_of_campaign_territories(a).isdisjoint(a["usage_rights"]["territories"])
    ]
    assert clashing == ["AST-015"]


async def test_seed_flaw_draft_among_finals_in_same_campaign(connect):
    async with connect() as client:
        drafts = await _search(client, status="draft")
        assert [a["id"] for a in drafts] == ["AST-009"]
        campaign = await _search(client, campaign_name=drafts[0]["campaign_name"])

    assert "final" in {a["status"] for a in campaign}


async def test_seed_flaw_channel_not_in_brands_approved_channels(connect):
    async with connect() as client:
        assets = await _search(client)
        approved = {}
        for brand in {a["brand"] for a in assets}:
            resource = await client.read_resource(f"brand://guidelines/{brand}")
            approved[brand] = json.loads(resource.contents[0].text)["approved_channels"]

    unapproved = [a["id"] for a in assets if a["channel"] not in approved[a["brand"]]]
    assert unapproved == ["AST-019"]
