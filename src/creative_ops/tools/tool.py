from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel


class ToolResultError(Exception):
    """Raised by a handler to give the model its message, verbatim, as an `is_error` result."""


@dataclass(frozen=True)
class Tool:
    """A tool the model can call; its input is checked against `input_model` first.

    `input_schema`, when set, is sent to the model in place of the one `input_model`
    generates: for a client-stack tool, whose schema the MCP server owns.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Awaitable[str]]
    input_schema: dict[str, Any] | None = None

    def to_param(self) -> ToolParam:
        """The tool definition sent to the model."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema or self.input_model.model_json_schema(),
        }
