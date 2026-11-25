import itertools
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import pytest
from anthropic.types import Message
from mcp import Client

from mcp_server.server import build_server


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> Message:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    """Stands in for AsyncAnthropic: returns scripted responses in order, records each call."""

    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


@pytest.fixture
def fake_client() -> Callable[..., FakeClient]:
    """Build a fake client from scripted responses: `fake_client(resp1, resp2, ...)`."""
    return lambda *responses: FakeClient(list(responses))


class RoutingMessages:
    def __init__(self, routes: dict[str, list[Message]]) -> None:
        self._routes = {prefix: list(responses) for prefix, responses in routes.items()}
        self.calls: dict[str, list[dict]] = {prefix: [] for prefix in routes}

    async def create(self, **kwargs) -> Message:
        system = kwargs.get("system")
        for prefix, responses in self._routes.items():
            if isinstance(system, str) and system.startswith(prefix):
                self.calls[prefix].append(kwargs)
                if not responses:
                    raise AssertionError(f"Route {prefix!r} has no scripted responses left")
                return responses.pop(0)
        raise AssertionError(f"No route for system prompt {system!r}")


class RoutingClient:
    """Stands in for AsyncAnthropic when several agents share one client.

    Each call goes to the scripted queue whose key its `system` prompt starts with, so
    concurrent Runs stay deterministic. Calls are recorded per route.
    """

    def __init__(self, routes: dict[str, list[Message]]) -> None:
        self.messages = RoutingMessages(routes)


@pytest.fixture
def routing_client() -> Callable[..., RoutingClient]:
    """Build a routing client: `routing_client({system_prefix: [resp1, ...], ...})`."""
    return RoutingClient


@pytest.fixture
def make_message() -> Callable[..., Message]:
    """Build an assistant `Message` with the given stop_reason, text, tool calls and token usage.

    Each entry in `tool_uses` is a `{"id", "name", "input"}` dict, emitted as a tool_use block.
    """

    def _make(
        stop_reason: str,
        text: str = "",
        input_tokens: int = 10,
        output_tokens: int = 5,
        tool_uses: Sequence[dict] = (),
    ) -> Message:
        content = [{"type": "text", "text": text}] if text else []
        content += [{"type": "tool_use", **tool_use} for tool_use in tool_uses]
        return Message(
            id="msg_test",
            type="message",
            role="assistant",
            model="test-model",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        )

    return _make


@pytest.fixture
def connect() -> Callable[[], Client]:
    """Open a real MCP client session, in memory, to a freshly built server.

    Use as `async with connect() as client:`. Entered inside the test rather than yielded
    from the fixture, because anyio requires the session to close in the task that opened it.
    """
    return lambda: Client(build_server())


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    """No test sends traces, even with Langfuse keys in the developer's `.env`."""
    monkeypatch.setattr("creative_ops.tracing._get_client", lambda: None)


_ids = itertools.count(1)


class FakeObservation:
    """One recorded observation: the fields it was opened with, updated, and its children."""

    def __init__(self, fields: dict) -> None:
        self.fields = dict(fields)
        self.children: list[FakeObservation] = []
        self.id = f"span{next(_ids)}"
        self.trace_id = "trace"

    def update(self, **fields) -> None:
        self.fields.update(fields)

    @property
    def name(self) -> str:
        return self.fields["name"]


class FakeLangfuse:
    """Stands in for the Langfuse client: records the tree of observations, nested by context."""

    def __init__(self) -> None:
        self.roots: list[FakeObservation] = []
        self.flushed = 0
        self._by_id: dict[str, FakeObservation] = {}
        self._current: ContextVar[FakeObservation | None] = ContextVar("current", default=None)

    @contextmanager
    def start_as_current_observation(self, **fields):
        observation = FakeObservation(fields)
        parent_id = (fields.get("trace_context") or {}).get("parent_span_id")
        parent = self._by_id[parent_id] if parent_id else self._current.get()
        self._by_id[observation.id] = observation
        (parent.children if parent else self.roots).append(observation)
        token = self._current.set(observation)
        try:
            yield observation
        finally:
            self._current.reset(token)

    def flush(self) -> None:
        self.flushed += 1


@pytest.fixture
def fake_langfuse(monkeypatch) -> FakeLangfuse:
    """Turn tracing on, recording into a fake client."""
    fake = FakeLangfuse()
    monkeypatch.setattr("creative_ops.tracing._get_client", lambda: fake)
    return fake
