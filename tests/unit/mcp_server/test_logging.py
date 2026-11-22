import json

import pytest


@pytest.mark.parametrize(
    ("arguments", "outcome"),
    [({"asset_id": "AST-001"}, "ok"), ({"asset_id": "AST-999"}, "error"), ({}, "error")],
)
async def test_tool_call_any_outcome_logs_one_json_line_to_stderr(
    connect, capsys, arguments, outcome
):
    async with connect() as client:
        capsys.readouterr()
        await client.call_tool("dam_get_asset", arguments)

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    log = json.loads(line)
    assert log["tool"] == "dam_get_asset"
    assert log["outcome"] == outcome
    assert isinstance(log["latency_ms"], float)
