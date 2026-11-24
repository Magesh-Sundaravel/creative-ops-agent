"""The app uvicorn serves: `uv run uvicorn creative_ops.api.main:app`."""

from anthropic import AsyncAnthropic

from creative_ops.agents.coordinator import coordinate, resume
from creative_ops.api.app import create_app
from creative_ops.config import get_settings
from creative_ops.tools.mcp_adapter import open_client_stack

app = create_app(
    coordinate,
    resume,
    open_client_stack,
    AsyncAnthropic(api_key=get_settings().anthropic_api_key.get_secret_value()),
)
