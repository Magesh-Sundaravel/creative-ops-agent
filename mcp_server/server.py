"""The `client_stack` MCP server, built fresh from seed data by `build_server`."""

import functools
import json
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, TypeAdapter

from mcp_server.models import Asset, Brand, ToolError

SEED_DIR = Path(__file__).parent / "seed"


def _load[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Read a JSON seed file and validate it as a list of `model`; raises on bad data."""
    return TypeAdapter(list[model]).validate_json(path.read_bytes())


def _ok(model: BaseModel) -> CallToolResult:
    """Wrap a successful result as structured content, with the same JSON as text."""
    return CallToolResult(
        content=[TextContent(type="text", text=model.model_dump_json())],
        structured_content=model.model_dump(mode="json"),
    )


def _not_found(message: str, **details: Any) -> CallToolResult:
    """An `isError` result carrying a non-retryable `not_found` ToolError as JSON."""
    error = ToolError(type="not_found", message=message, retryable=False, details=details)
    return CallToolResult(
        content=[TextContent(type="text", text=error.model_dump_json())], is_error=True
    )


def _logged[**P](
    tool: Callable[P, Awaitable[CallToolResult]],
) -> Callable[P, Awaitable[CallToolResult]]:
    """Write one JSON line to stderr per call: tool name, latency and ok or error.

    stderr, because stdout carries the stdio protocol.
    """

    @functools.wraps(tool)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> CallToolResult:
        start = time.perf_counter()
        outcome = "error"
        try:
            result = await tool(*args, **kwargs)
            outcome = "error" if result.is_error else "ok"
            return result
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log = {"tool": tool.__name__, "latency_ms": latency_ms, "outcome": outcome}
            print(json.dumps(log), file=sys.stderr)

    return wrapper


def build_server(seed_dir: Path = SEED_DIR) -> MCPServer:
    """Build a new server whose state starts from the seed files in `seed_dir`."""
    _load(seed_dir / "brands.json", Brand)  # validated now; served from the brand resources
    assets = {asset.id: asset for asset in _load(seed_dir / "assets.json", Asset)}

    server = MCPServer("client_stack")

    @server.tool()
    @_logged
    async def dam_get_asset(asset_id: str) -> Annotated[CallToolResult, Asset]:
        """Fetch one Asset from the DAM by its id, with full metadata and Usage Rights.

        Use this when you already know the Asset id (for example from a search result).
        To find Assets by brand, campaign or channel, use `dam_search_assets` instead.

        Args:
            asset_id: The DAM Asset id, e.g. "AST-001".
        """
        if asset_id not in assets:
            return _not_found(
                f"No Asset with id {asset_id!r}. Check the id, or find Assets with "
                "dam_search_assets.",
                asset_id=asset_id,
            )
        return _ok(assets[asset_id])

    return server
