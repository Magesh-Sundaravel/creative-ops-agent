"""The `client_stack` MCP server, built fresh from seed data by `build_server`."""

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as SDKToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent
from pydantic import BaseModel, TypeAdapter, ValidationError

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


def _error(type: str, message: str, **details: Any) -> CallToolResult:
    """An `isError` result carrying a ToolError as JSON."""
    error = ToolError(
        type=type, message=message, retryable=type == "upstream_unavailable", details=details
    )
    return CallToolResult(
        content=[TextContent(type="text", text=error.model_dump_json())], is_error=True
    )


class _ClientStack(MCPServer):
    """MCPServer that logs every tool call and reports bad arguments as a ToolError."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult | InputRequiredResult:
        """Run the tool, then write one JSON line to stderr: tool, latency and ok or error.

        stderr, because stdout carries the stdio protocol.
        """
        start = time.perf_counter()
        outcome = "error"
        try:
            try:
                result = await super().call_tool(name, arguments, context)
            except SDKToolError as exc:
                if not isinstance(exc.__cause__, ValidationError):
                    raise
                errors = [
                    {"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]}
                    for e in exc.__cause__.errors()
                ]
                problems = "; ".join(f"{e['field']}: {e['problem']}" for e in errors)
                result = _error(
                    "invalid_input",
                    f"Invalid arguments for {name} ({problems}). Fix them and call again.",
                    errors=errors,
                )
            outcome = "error" if getattr(result, "is_error", False) else "ok"
            return result
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log = {"tool": name, "latency_ms": latency_ms, "outcome": outcome}
            print(json.dumps(log), file=sys.stderr)


def build_server(seed_dir: Path = SEED_DIR) -> MCPServer:
    """Build a new server whose state starts from the seed files in `seed_dir`.

    Raises if the seed fails validation, so bad seed data stops the server from starting.
    """
    brand_slugs = {brand.slug for brand in _load(seed_dir / "brands.json", Brand)}
    assets = {asset.id: asset for asset in _load(seed_dir / "assets.json", Asset)}
    for asset in assets.values():
        if asset.brand not in brand_slugs:
            raise ValueError(f"Seed Asset {asset.id} has unknown brand {asset.brand!r}")

    server = _ClientStack("client_stack")

    @server.tool()
    async def dam_get_asset(asset_id: str) -> Annotated[CallToolResult, Asset]:
        """Fetch one Asset from the DAM by its id, with full metadata and Usage Rights.

        Use this when you already know the Asset id (for example from a search result).
        To find Assets by brand, campaign or channel, use `dam_search_assets` instead.

        Args:
            asset_id: The DAM Asset id, e.g. "AST-001".
        """
        if asset_id not in assets:
            return _error(
                "not_found",
                f"No Asset with id {asset_id!r}. Check the id, or find Assets with "
                "dam_search_assets.",
                asset_id=asset_id,
            )
        return _ok(assets[asset_id])

    return server
