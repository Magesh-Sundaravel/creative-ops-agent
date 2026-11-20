from collections.abc import Callable, Sequence

import pytest
from anthropic.types import Message


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
