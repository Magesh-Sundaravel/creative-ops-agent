"""The `client_stack` MCP server, built fresh from seed data by `build_server`."""

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Annotated, Any, get_args

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.server.mcpserver.exceptions import ToolError as SDKToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent
from pydantic import BaseModel, TypeAdapter, ValidationError

from mcp_server.models import Asset, AssetList, Brand, Task, TaskStatus, ToolError

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
    brands = {brand.slug: brand for brand in _load(seed_dir / "brands.json", Brand)}
    brand_slugs = brands.keys()
    assets = {asset.id: asset for asset in _load(seed_dir / "assets.json", Asset)}
    for asset in assets.values():
        if asset.brand not in brand_slugs:
            raise ValueError(f"Seed Asset {asset.id} has unknown brand {asset.brand!r}")
    tasks: dict[str, Task] = {}
    for task in _load(seed_dir / "tasks.json", Task):
        if task.id in tasks:
            raise ValueError(f"Seed Tasks share the id {task.id!r}")
        if task.brand not in brand_slugs:
            raise ValueError(f"Seed Task {task.id} has unknown brand {task.brand!r}")
        tasks[task.id] = task
    next_task_number = max((int(task_id.removeprefix("TSK-")) for task_id in tasks), default=0) + 1

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

    @server.tool()
    async def dam_search_assets(
        brand: str | None = None,
        campaign_name: str | None = None,
        asset_type: str | None = None,
        channel: str | None = None,
        status: str | None = None,
    ) -> Annotated[CallToolResult, AssetList]:
        """Find Assets in the DAM by metadata. Every filter is optional.

        Filters are exact matches ignoring case, combined with AND. No filters returns every
        Asset, and a search that matches nothing returns an empty list. There is no free-text
        query. Use `dam_get_asset` to fetch one Asset whose id you already know.

        Args:
            brand: Brand slug, e.g. "loomhouse". Must be a known brand.
            campaign_name: Campaign name, e.g. "Spring Linen Edit".
            asset_type: e.g. "image" or "video".
            channel: e.g. "instagram", "tiktok" or "website".
            status: "draft" or "final".
        """
        if brand is not None and brand.casefold() not in brand_slugs:
            return _error(
                "not_found",
                f"No brand with slug {brand!r}. Known brands: {', '.join(sorted(brand_slugs))}.",
                brand=brand,
            )
        filters = {
            "brand": brand,
            "campaign_name": campaign_name,
            "asset_type": asset_type,
            "channel": channel,
            "status": status,
        }
        wanted = {field: value.casefold() for field, value in filters.items() if value is not None}
        found = [
            asset
            for asset in assets.values()
            if all(getattr(asset, field).casefold() == value for field, value in wanted.items())
        ]
        return _ok(AssetList(assets=found))

    @server.tool()
    async def tracker_create_task(
        title: str,
        brand: str,
        campaign_name: str,
        description: str,
        due_date: date | None = None,
    ) -> Annotated[CallToolResult, Task]:
        """Create a Task in the project tracker. It always starts with status "briefed".

        The tracker assigns the Task id; it is in the returned Task. Use
        `tracker_update_status` to move the Task on afterwards.

        Args:
            title: Short name for the work, e.g. "Spring Linen Edit: Instagram feed set".
            brand: Brand slug, e.g. "loomhouse", matched ignoring case. Must be a known brand.
            campaign_name: Campaign the work belongs to, e.g. "Spring Linen Edit".
            description: What production has to make.
            due_date: Optional due date, ISO format (YYYY-MM-DD).
        """
        nonlocal next_task_number
        if brand.casefold() not in brand_slugs:
            return _error(
                "not_found",
                f"No brand with slug {brand!r}. Known brands: {', '.join(sorted(brand_slugs))}.",
                brand=brand,
            )
        task = Task(
            id=f"TSK-{next_task_number:03d}",
            title=title,
            brand=brand.casefold(),
            campaign_name=campaign_name,
            description=description,
            due_date=due_date,
            status="briefed",
        )
        next_task_number += 1
        tasks[task.id] = task
        return _ok(task)

    @server.tool()
    async def tracker_update_status(task_id: str, status: str) -> Annotated[CallToolResult, Task]:
        """Set a Task's status in the project tracker and return the updated Task.

        Any status may follow any other. The status is matched ignoring case.

        Args:
            task_id: The Task id, e.g. "TSK-001".
            status: One of "briefed", "in_production", "in_review", "revisions_requested",
                "approved" or "delivered".
        """
        if task_id not in tasks:
            return _error(
                "not_found",
                f"No Task with id {task_id!r}. Check the id returned by tracker_create_task.",
                task_id=task_id,
            )
        valid_statuses = list(get_args(TaskStatus))
        if status.casefold() not in valid_statuses:
            return _error(
                "invalid_input",
                f"Invalid status {status!r}. Valid statuses: {', '.join(valid_statuses)}.",
                status=status,
                valid_statuses=valid_statuses,
            )
        tasks[task_id] = tasks[task_id].model_copy(update={"status": status.casefold()})
        return _ok(tasks[task_id])

    @server.resource("brand://brands", mime_type="application/json")
    async def list_brands() -> str:
        """Every known brand's slug and display name.

        Use it to map a brand name from a Brief to the slug the other tools and resources take.
        """
        return json.dumps(
            [brand.model_dump(include={"slug", "display_name"}) for brand in brands.values()]
        )

    @server.resource("brand://guidelines/{brand}", mime_type="application/json")
    async def brand_guidelines(brand: str) -> str:
        """A brand's Brand Guidelines: palette, typography, logo rules, tone of voice, do's,
        don'ts, mandatory legal lines and approved channels.

        `brand` is the brand slug, e.g. "loomhouse". List the slugs with `brand://brands`.
        """
        if brand not in brands:
            raise ResourceNotFoundError(
                f"No brand with slug {brand!r}. Known brands: {', '.join(sorted(brand_slugs))}."
            )
        return brands[brand].guidelines.model_dump_json()

    return server
