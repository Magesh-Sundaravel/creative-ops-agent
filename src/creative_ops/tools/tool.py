from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    """A tool the model can call; its input is checked against `input_model` first."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Awaitable[str]]

    def to_param(self) -> ToolParam:
        """The tool definition sent to the model, with the schema generated from `input_model`."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }
