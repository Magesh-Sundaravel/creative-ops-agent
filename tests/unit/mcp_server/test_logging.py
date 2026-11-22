import json

import pytest


@pytest.mark.parametrize(
    ("asset_id", "outcome"),
    [("AST-001", "ok"), ("AST-999", "error")],
)
async def test_tool_call_any_outcome_logs_one_json_line_to_stderr(
    connect, capsys, asset_id, outcome
):
    async with connect() as client:
        capsys.readouterr()
        await client.call_tool("dam_get_asset", {"asset_id": asset_id})

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    log = json.loads(line)
    assert log["tool"] == "dam_get_asset"
    assert log["outcome"] == outcome
    assert isinstance(log["latency_ms"], float)
