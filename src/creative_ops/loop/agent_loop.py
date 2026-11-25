"""Hand-written, stop_reason-driven agent loop (see ADR 0001)."""

import asyncio
import logging
from collections.abc import Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Any

from anthropic import AsyncAnthropic, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock
from pydantic import BaseModel, ValidationError

from creative_ops import tracing
from creative_ops.config import get_settings
from creative_ops.pricing import cost_usd
from creative_ops.tools.tool import Tool, ToolResultError

logger = logging.getLogger(__name__)


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
    """Tokens consumed by a Run, and what they cost (None if any call was on an unpriced model)."""

    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None = Decimal(0)


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
    except ValidationError as exc:
        # Field and message only: no rejected values or docs URLs echoed back to the model.
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_url=False)
        )
        return error(f"Invalid input for tool {block.name!r}: {problems}")
    except Exception as exc:  # validators may raise more than ValidationError
        return error(f"Invalid input for tool {block.name!r}: {type(exc).__name__}: {exc}")
    try:
        output = await tool.handler(args)
    except ToolResultError as exc:
        return error(str(exc))
    except Exception as exc:
        logger.exception("Tool %r raised", block.name)
        return error(f"Tool {block.name!r} failed: {type(exc).__name__}: {exc}")
    if not isinstance(output, str):
        return error(f"Tool {block.name!r} returned {type(output).__name__}, expected str")
    return {"type": "tool_result", "tool_use_id": block.id, "content": output}


async def _traced_call(
    block: ToolUseBlock, tools_by_name: dict[str, Tool], parent: dict[str, str] | None
) -> ToolResultBlockParam:
    """Run one tool call inside its own span under the requesting Iteration."""
    with tracing.tool_span(block.name, block.input, parent) as span:
        result = await _call_tool(block, tools_by_name)
        is_error = result.get("is_error", False)
        span.update(output=result["content"], metadata={"is_error": is_error})
        if is_error:
            span.update(level="ERROR", status_message=result["content"])
        return result


def _log_run(stop_reason: str | None, usage: TokenUsage) -> None:
    """One line per Run: how it stopped, its tokens and its cost (n/a for an unpriced model)."""
    logger.info(
        "Run stopped (%s): input_tokens=%d output_tokens=%d cost=%s",
        stop_reason,
        usage.input_tokens,
        usage.output_tokens,
        "n/a" if usage.cost_usd is None else f"${usage.cost_usd:.6f}",
    )


async def run(
    client: AsyncAnthropic,
    prompt: str,
    tools: Sequence[Tool],
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 5,
    agent: str = "agent",
) -> LoopResult:
    """Drive a Run from a single prompt until the model stops.

    `model` defaults to the configured fast model, resolved here rather than at import.
    `agent` names the Run's span in the Trace.
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
    with tracing.run_span(agent, prompt, system, model) as run_span:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        for iteration in range(1, max_iterations + 1):
            with tracing.iteration_span(model, list(messages), system, iteration) as generation:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system if system is not None else omit,
                    messages=list(messages),
                    tools=[tool.to_param() for tool in tools] if tools else omit,
                )
                call_cost = cost_usd(
                    model, response.usage.input_tokens, response.usage.output_tokens
                )
                generation.update(
                    output=tracing.assistant_message(response.content),
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                    cost_details=None if call_cost is None else {"total": float(call_cost)},
                    metadata={
                        "iteration": iteration,
                        "stop_reason": response.stop_reason,
                        "thinking": tracing.thinking(response.content) or None,
                    },
                )
            iteration_context = generation.context
            messages.append({"role": "assistant", "content": response.content})
            usage = TokenUsage(
                input_tokens=usage.input_tokens + response.usage.input_tokens,
                output_tokens=usage.output_tokens + response.usage.output_tokens,
                cost_usd=None
                if usage.cost_usd is None or call_cost is None
                else usage.cost_usd + call_cost,
            )
            if response.stop_reason != "tool_use":
                break
            if iteration == max_iterations:
                _log_run(response.stop_reason, usage)
                raise MaxIterationsExceeded(max_iterations, messages, usage)
            results: list[ToolResultBlockParam] = await asyncio.gather(
                *(
                    _traced_call(block, tools_by_name, iteration_context)
                    for block in response.content
                    if block.type == "tool_use"
                )
            )
            if not results:
                _log_run(response.stop_reason, usage)
                raise UnexpectedStopReason(response.stop_reason, messages, usage)
            messages.append({"role": "user", "content": results})
        _log_run(response.stop_reason, usage)
        if response.stop_reason not in _OUTCOMES:
            raise UnexpectedStopReason(response.stop_reason, messages, usage)
        text = "".join(block.text for block in response.content if block.type == "text")
        outcome = _OUTCOMES[response.stop_reason]
        run_span.update(
            output=text,
            metadata={
                "agent": agent,
                "outcome": outcome,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": None if usage.cost_usd is None else float(usage.cost_usd),
            },
        )
        return LoopResult(text=text, outcome=outcome, messages=messages, usage=usage)
