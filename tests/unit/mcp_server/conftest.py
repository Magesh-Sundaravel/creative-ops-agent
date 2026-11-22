from collections.abc import Callable

import pytest
from mcp import Client

from mcp_server.server import build_server


@pytest.fixture
def connect() -> Callable[[], Client]:
    """Open a real MCP client session, in memory, to a freshly built server.

    Use as `async with connect() as client:`. Entered inside the test rather than yielded
    from the fixture, because anyio requires the session to close in the task that opened it.
    """
    return lambda: Client(build_server())
