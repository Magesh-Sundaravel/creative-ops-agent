import subprocess
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

REPO_ROOT = Path(__file__).parents[3]


async def test_main_over_stdio_serves_dam_get_asset():
    """`python -m mcp_server` speaks MCP on stdout; any stray stdout write would break this."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], cwd=REPO_ROOT)

    async with Client(params) as client:
        result = await client.call_tool("dam_get_asset", {"asset_id": "AST-001"})

    assert not result.is_error
    assert result.structured_content["id"] == "AST-001"


def test_main_startup_and_shutdown_write_nothing_to_stdout():
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server"],
        input=b"",
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b""
