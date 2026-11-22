"""Run the `client_stack` server over stdio: `uv run python -m mcp_server`."""

from mcp_server.server import build_server

build_server().run()
