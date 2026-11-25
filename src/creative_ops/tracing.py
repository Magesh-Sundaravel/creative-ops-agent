"""Langfuse tracing: the only module that talks to the SDK (ADR 0006).

With no keys set, every function here does nothing. An SDK error is logged and swallowed,
so tracing can never fail a Run.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any

from langfuse import Langfuse
from langfuse.span_filter import is_default_export_span

from creative_ops.config import get_settings

logger = logging.getLogger(__name__)

ANTHROPIC_SDK_SCOPE = "com.anthropic.sdk.python"

_client: Langfuse | None = None
_client_built = False


def _should_export(span: Any) -> bool:
    """Export LLM spans, except the Anthropic SDK's own: each Iteration already records its call.

    Left in, every model call would show up twice and its cost would be counted twice.
    """
    scope = span.instrumentation_scope
    return (scope is None or scope.name != ANTHROPIC_SDK_SCOPE) and is_default_export_span(span)


def _get_client() -> Langfuse | None:
    """The Langfuse client, or None when tracing is off (either key unset)."""
    global _client, _client_built
    if not _client_built:
        _client_built = True
        try:
            settings = get_settings()
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                _client = Langfuse(
                    public_key=settings.langfuse_public_key.get_secret_value(),
                    secret_key=settings.langfuse_secret_key.get_secret_value(),
                    base_url=settings.langfuse_host,
                    should_export_span=_should_export,
                )
        except Exception:
            logger.warning("Tracing is off: could not set up the Langfuse client", exc_info=True)
    return _client


class Span:
    """A handle on one observation; updating it never raises, and is a no-op when tracing is off."""

    def __init__(self, observation: Any | None) -> None:
        """Wrap `observation`, or nothing when tracing is off."""
        self._observation = observation

    def update(self, **fields: Any) -> None:
        """Set fields (output, metadata, usage_details, ...) on the observation."""
        if self._observation is None:
            return
        try:
            self._observation.update(**fields)
        except Exception:
            logger.warning("Could not update a Langfuse observation", exc_info=True)

    @property
    def context(self) -> dict[str, str] | None:
        """Where to nest a child under this observation after it has closed, or None."""
        if self._observation is None:
            return None
        try:
            return {"trace_id": self._observation.trace_id, "parent_span_id": self._observation.id}
        except Exception:
            logger.warning("Could not read a Langfuse observation's id", exc_info=True)
            return None


@contextmanager
def _observe(**fields: Any) -> Iterator[Span]:
    """Open an observation that is the current parent for the block; mark it on an exception."""
    client = _get_client()
    stack = ExitStack()
    observation = None
    if client is not None:
        try:
            observation = stack.enter_context(client.start_as_current_observation(**fields))
        except Exception:
            logger.warning("Could not open a Langfuse observation", exc_info=True)
    span = Span(observation)
    try:
        yield span
    except Exception as exc:
        span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            stack.close()
        except Exception:
            logger.warning("Could not close a Langfuse observation", exc_info=True)


def _block(block: Any) -> dict[str, Any]:
    """A content block as a dict, whether the SDK gave an object or the loop a dict."""
    return block if isinstance(block, dict) else block.model_dump(exclude_none=True)


def assistant_message(content: list[Any]) -> dict[str, Any]:
    """Model output as an OpenAI-style assistant message, which Langfuse renders with tool calls."""
    blocks = [_block(b) for b in content]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(b["text"] for b in blocks if b["type"] == "text"),
    }
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
        }
        for b in blocks
        if b["type"] == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def thinking(content: list[Any]) -> str:
    """The model's thinking in a response, without the opaque signature."""
    return "\n\n".join(b["thinking"] for b in map(_block, content) if b["type"] == "thinking")


def chat(messages: list[Any]) -> list[dict[str, Any]]:
    """Anthropic messages as role-labeled chat: tool results become `tool` messages by call id.

    Thinking blocks are left out; the generation that produced them records them.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            out.append(assistant_message(content))
        else:
            out += [
                {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                for b in map(_block, content)
                if b["type"] == "tool_result"
            ]
    return out


def _with_system(system: str | None, messages: list[Any]) -> list[Any]:
    """The prompt as a chat, which Langfuse renders by role."""
    return messages if system is None else [{"role": "system", "content": system}, *messages]


def trace_brief(brief_id: str, brief: str):
    """The root of a Brief's Trace; the Brief id is the trace id, so a resume continues it."""
    return _observe(
        as_type="span", name="process-brief", trace_context={"trace_id": brief_id}, input=brief
    )


def run_span(agent: str, prompt: str, system: str | None, model: str):
    """One Run, nested under whatever is current (the Brief, or the Coordinator's Run)."""
    return _observe(
        as_type="agent",
        name=f"{agent}-run",
        input=_with_system(system, [{"role": "user", "content": prompt}]),
        metadata={"agent": agent, "model": model},
    )


def iteration_span(model: str, messages: list[Any], system: str | None, number: int):
    """One Iteration: a model call, recorded as a generation."""
    return _observe(
        as_type="generation",
        name="call-model",
        model=model,
        input=_with_system(system, chat(messages)),
        metadata={"iteration": number},
    )


def tool_span(
    name: str, tool_input: Any, parent: dict[str, str] | None
) -> AbstractContextManager[Span]:
    """One tool call, nested under the Iteration `parent` that requested it.

    The Iteration's own span is closed by the time its tools run, and concurrent calls
    share one context, so the parent is passed explicitly.
    """
    fields: dict[str, Any] = {"as_type": "tool", "name": name, "input": tool_input}
    if parent is not None:
        fields["trace_context"] = parent
    return _observe(**fields)


def flush() -> None:
    """Send what is still queued; call before the process exits."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.warning("Could not flush Langfuse", exc_info=True)
