"""Access to the client stack over an open MCP session: from code, and as model `Tool`s."""

import sys
from collections.abc import Collection
from pathlib import Path

from mcp import Client, StdioServerParameters
from pydantic import BaseModel, ConfigDict, TypeAdapter

from creative_ops.tools.tool import Tool, ToolResultError

REPO_ROOT = Path(__file__).parents[3]


class ClientStackError(RuntimeError):
    """A client-stack tool returned an error result; the message is the server's ToolError JSON."""

    def __init__(self, tool: str, error: str) -> None:
        super().__init__(f"{tool} failed: {error}")
        self.tool = tool
        self.error = error


def open_client_stack() -> Client:
    """A client for one Run's session to `python -m mcp_server` over stdio.

    Use as `async with open_client_stack() as session:`.
    """
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], cwd=REPO_ROOT)
    return Client(params)


async def call_tool[T: BaseModel](
    session: Client, name: str, arguments: dict, result_model: type[T]
) -> T:
    """Call a client-stack tool and validate its structured result as `result_model`."""
    result = await session.call_tool(name, arguments)
    if result.is_error:
        text = "".join(block.text for block in result.content if block.type == "text")
        raise ClientStackError(name, text)
    return result_model.model_validate(result.structured_content)


async def read_resource[T](session: Client, uri: str, adapter: TypeAdapter[T]) -> T:
    """Read a JSON resource and validate it with `adapter`."""
    result = await session.read_resource(uri)
    [content] = result.contents
    return adapter.validate_json(content.text)


class _ServerArgs(BaseModel):
    """Passes the model's arguments through unchanged; the server validates them."""

    model_config = ConfigDict(extra="allow")


def _tool(session: Client, name: str, description: str, input_schema: dict) -> Tool:
    """A loop `Tool` that calls the server tool `name`; a server error becomes `is_error`."""

    async def handler(args: _ServerArgs) -> str:
        result = await session.call_tool(name, args.model_dump())
        text = "".join(block.text for block in result.content if block.type == "text")
        if result.is_error:
            raise ToolResultError(text)
        return text

    return Tool(
        name=name,
        description=description,
        input_model=_ServerArgs,
        handler=handler,
        input_schema=input_schema,
    )


async def mcp_tools(session: Client, allowlist: Collection[str]) -> list[Tool]:
    """The server tools named in `allowlist`, as loop `Tool`s with the server's input schemas.

    Raises if the server has no tool by one of the names.
    """
    listed = {tool.name: tool for tool in (await session.list_tools()).tools}
    missing = sorted(set(allowlist) - listed.keys())
    if missing:
        raise ValueError(f"Client stack has no tools {missing}")
    return [
        _tool(session, name, listed[name].description or "", listed[name].input_schema)
        for name in allowlist
    ]
