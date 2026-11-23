from creative_ops.schemas.client_stack import Asset
from creative_ops.tools.mcp_adapter import call_tool, open_client_stack


async def test_open_client_stack_serves_dam_get_asset_over_stdio():
    async with open_client_stack() as session:
        asset = await call_tool(session, "dam_get_asset", {"asset_id": "AST-001"}, Asset)

    assert asset.id == "AST-001"
