"""Hand-written, stop_reason-driven agent loop (see ADR 0001)."""

from enum import StrEnum
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import BaseModel

from creative_ops.config import get_settings


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


class UnexpectedStopReason(RuntimeError):
    """The model stopped for a reason the loop does not handle."""


class TokenUsage(BaseModel):
    """Tokens consumed by a Run."""

    input_tokens: int
    output_tokens: int


class LoopResult(BaseModel):
    """The outcome of a Run."""

    text: str
    outcome: StopOutcome
    messages: list[dict[str, Any]]  # MessageParam dicts; content may hold SDK blocks
    usage: TokenUsage


async def run(
    client: AsyncAnthropic,
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
) -> LoopResult:
    """Drive a Run from a single prompt until the model stops.

    `model` defaults to the configured fast model, resolved here rather than at import.
    """
    if model is None:
        model = get_settings().model_fast
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    request: dict = {"model": model, "max_tokens": max_tokens}
    if system is not None:
        request["system"] = system
    response = await client.messages.create(**request, messages=list(messages))
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason not in _OUTCOMES:
        raise UnexpectedStopReason(f"Unhandled stop_reason: {response.stop_reason!r}")
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = TokenUsage(
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
    )
    return LoopResult(
        text=text, outcome=_OUTCOMES[response.stop_reason], messages=messages, usage=usage
    )
