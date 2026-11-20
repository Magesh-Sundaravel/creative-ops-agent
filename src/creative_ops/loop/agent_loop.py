"""Hand-written, stop_reason-driven agent loop (see ADR 0001)."""

import asyncio
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from anthropic import AsyncAnthropic, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock
from pydantic import BaseModel

from creative_ops.config import get_settings
from creative_ops.tools.tool import Tool


class StopOutcome(StrEnum):
    """Why a Run ended."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    REFUSED = "refused"


_OUTCOMES: dict[str, StopOutcome] = {
    "end_turn": StopOutcome.COMPLETED,
    "stop_sequence": StopOutcome.COMPLETED,
    "max_tokens": StopOutcome.TRUNCATED,
    "refusal": StopOutcome.REFUSED,
}


class TokenUsage(BaseModel):
    """Tokens consumed by a Run."""

    input_tokens: int
    output_tokens: int


class UnexpectedStopReason(RuntimeError):
    """The model stopped for a reason the loop does not handle.

    Carries the Run's trace so far, so a failed Run can still be inspected.
    """

    def __init__(
        self, stop_reason: str | None, messages: list[MessageParam], usage: TokenUsage
    ) -> None:
        super().__init__(f"Unhandled stop_reason: {stop_reason!r}")
        self.stop_reason = stop_reason
        self.messages = messages
        self.usage = usage


class MaxIterationsExceeded(RuntimeError):
    """The model still wanted tools when the Run reached its Iteration cap.

    Carries the Run's trace so far, so a runaway Run can still be inspected.
    """

    def __init__(
        self, max_iterations: int, messages: list[MessageParam], usage: TokenUsage
    ) -> None:
        super().__init__(f"Run exceeded {max_iterations} Iterations")
        self.max_iterations = max_iterations
        self.messages = messages
        self.usage = usage


class LoopResult(BaseModel):
    """The outcome of a Run."""

    text: str
    outcome: StopOutcome
    messages: list[dict[str, Any]]  # MessageParam dicts; content may hold SDK blocks
    usage: TokenUsage


async def _call_tool(block: ToolUseBlock, tools_by_name: dict[str, Tool]) -> ToolResultBlockParam:
    """Run one tool call; a failure becomes an `is_error` result the model can see."""

    def error(message: str) -> ToolResultBlockParam:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": message,
            "is_error": True,
        }

    tool = tools_by_name.get(block.name)
    if tool is None:
        return error(f"Unknown tool {block.name!r}. Available tools: {sorted(tools_by_name)}")
    try:
        args = tool.input_model.model_validate(block.input)
    except Exception as exc:  # validators may raise more than ValidationError
        return error(f"Invalid input for tool {block.name!r}: {exc}")
    try:
        output = await tool.handler(args)
    except Exception as exc:
        return error(f"Tool {block.name!r} failed: {type(exc).__name__}: {exc}")
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


async def run(
    client: AsyncAnthropic,
    prompt: str,
    tools: Sequence[Tool],
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 5,
) -> LoopResult:
    """Drive a Run from a single prompt until the model stops.

    `model` defaults to the configured fast model, resolved here rather than at import.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    tools_by_name = {tool.name: tool for tool in tools}
    if len(tools_by_name) != len(tools):
        names = [tool.name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate tool names: {duplicates}")
    if model is None:
        model = get_settings().model_fast
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    usage = TokenUsage(input_tokens=0, output_tokens=0)
    for iteration in range(1, max_iterations + 1):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system if system is not None else omit,
            messages=list(messages),
            tools=[tool.to_param() for tool in tools] if tools else omit,
        )
        messages.append({"role": "assistant", "content": response.content})
        usage = TokenUsage(
            input_tokens=usage.input_tokens + response.usage.input_tokens,
            output_tokens=usage.output_tokens + response.usage.output_tokens,
        )
        if response.stop_reason != "tool_use":
            break
        if iteration == max_iterations:
            raise MaxIterationsExceeded(max_iterations, messages, usage)
        results: list[ToolResultBlockParam] = await asyncio.gather(
            *(
                _call_tool(block, tools_by_name)
                for block in response.content
                if block.type == "tool_use"
            )
        )
        if not results:
            raise UnexpectedStopReason(response.stop_reason, messages, usage)
        messages.append({"role": "user", "content": results})
    if response.stop_reason not in _OUTCOMES:
        raise UnexpectedStopReason(response.stop_reason, messages, usage)
    text = "".join(block.text for block in response.content if block.type == "text")
    return LoopResult(
        text=text, outcome=_OUTCOMES[response.stop_reason], messages=messages, usage=usage
    )
