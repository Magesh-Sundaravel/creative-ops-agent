"""Code-side access to the client stack over an open MCP session."""

import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from pydantic import BaseModel, TypeAdapter

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
