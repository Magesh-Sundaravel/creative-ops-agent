import json
import os
import subprocess
import sys

import pytest
from anthropic import omit
from pydantic import BaseModel

from creative_ops.config import get_settings
from creative_ops.loop.agent_loop import (
    MaxIterationsExceeded,
    StopOutcome,
    TokenUsage,
    UnexpectedStopReason,
    run,
)
from creative_ops.tools.tool import Tool

MODEL = "test-model"


class AddInput(BaseModel):
    a: int
    b: int


def make_add_tool(received: list[AddInput] | None = None) -> Tool:
    """The dummy `add` tool; records each validated input it receives."""

    async def handler(args: AddInput) -> str:
        if received is not None:
            received.append(args)
        return str(args.a + args.b)

    return Tool(name="add", description="Add two integers.", input_model=AddInput, handler=handler)


async def test_run_end_turn_returns_completed_with_text(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="Hello!"))

    result = await run(client, "Say hello", [], model=MODEL, max_tokens=256)

    assert result.outcome is StopOutcome.COMPLETED
    assert result.text == "Hello!"
    [call] = client.messages.calls
    assert call["model"] == MODEL
    assert call["max_tokens"] == 256
    assert call["messages"] == [{"role": "user", "content": "Say hello"}]


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", StopOutcome.COMPLETED),
        ("stop_sequence", StopOutcome.COMPLETED),
        ("max_tokens", StopOutcome.TRUNCATED),
        ("refusal", StopOutcome.REFUSED),
    ],
)
async def test_run_stop_reason_maps_to_outcome(fake_client, make_message, stop_reason, expected):
    client = fake_client(make_message(stop_reason, text="partial"))

    result = await run(client, "Go", [], model=MODEL)

    assert result.outcome is expected


async def test_run_unrecognised_stop_reason_raises(fake_client, make_message):
    client = fake_client(make_message("pause_turn"))

    with pytest.raises(UnexpectedStopReason, match="pause_turn"):
        await run(client, "Go", [], model=MODEL)


async def test_run_unrecognised_stop_reason_error_carries_trace(fake_client, make_message):
    reply = make_message("pause_turn", text="thinking", input_tokens=12, output_tokens=7)
    client = fake_client(reply)

    with pytest.raises(UnexpectedStopReason) as exc_info:
        await run(client, "Go", [], model=MODEL)

    error = exc_info.value
    assert error.stop_reason == "pause_turn"
    assert error.messages == [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": reply.content},
    ]
    assert error.usage == TokenUsage(input_tokens=12, output_tokens=7)


async def test_run_with_system_sends_system_prompt(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", [], model=MODEL, system="You are terse.")

    assert client.messages.calls[0]["system"] == "You are terse."


async def test_run_without_system_omits_system_param(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", [], model=MODEL)

    assert client.messages.calls[0]["system"] is omit


async def test_run_completed_returns_full_message_history(fake_client, make_message):
    reply = make_message("end_turn", text="Hi there")
    client = fake_client(reply)

    result = await run(client, "Say hi", [], model=MODEL)

    assert result.messages == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": reply.content},
    ]


async def test_run_completed_returns_token_usage(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok", input_tokens=12, output_tokens=7))

    result = await run(client, "Go", [], model=MODEL)

    assert result.usage == TokenUsage(input_tokens=12, output_tokens=7)


async def test_run_without_model_uses_configured_fast_model(fake_client, make_message, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("MODEL_FAST", "fast-test-model")
    get_settings.cache_clear()
    client = fake_client(make_message("end_turn", text="ok"))

    try:
        await run(client, "Go", [])
    finally:
        get_settings.cache_clear()

    assert client.messages.calls[0]["model"] == "fast-test-model"


def test_agent_loop_import_without_api_key_succeeds(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    proc = subprocess.run(
        [sys.executable, "-c", "import creative_ops.loop.agent_loop"],
        cwd=tmp_path,  # no .env here
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr


@pytest.mark.filterwarnings("error")
async def test_run_result_serializes_to_json_without_warnings(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="Hi there"))

    result = await run(client, "Say hi", [], model=MODEL)

    dumped = json.loads(result.model_dump_json())
    assert dumped["messages"][1]["content"][0]["text"] == "Hi there"


async def test_run_sends_tool_definitions_with_pydantic_schema(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", [make_add_tool()], model=MODEL)

    assert client.messages.calls[0]["tools"] == [
        {
            "name": "add",
            "description": "Add two integers.",
            "input_schema": {
                "type": "object",
                "title": "AddInput",
                "properties": {
                    "a": {"type": "integer", "title": "A"},
                    "b": {"type": "integer", "title": "B"},
                },
                "required": ["a", "b"],
            },
        }
    ]


ADD_CALL = {"id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}}


async def test_run_tool_use_calls_handler_and_sends_tool_result(fake_client, make_message):
    first = make_message("tool_use", text="Let me add.", tool_uses=[ADD_CALL])
    client = fake_client(first, make_message("end_turn", text="It's 5."))
    received: list[AddInput] = []

    await run(client, "What is 2 + 3?", [make_add_tool(received)], model=MODEL)

    assert received == [AddInput(a=2, b=3)]
    second_call = client.messages.calls[1]
    assert second_call["messages"] == [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "5"}],
        },
    ]


async def test_run_after_tool_round_trip_completes_with_full_history(fake_client, make_message):
    first = make_message("tool_use", tool_uses=[ADD_CALL])
    last = make_message("end_turn", text="It's 5.")
    client = fake_client(first, last)

    result = await run(client, "What is 2 + 3?", [make_add_tool()], model=MODEL)

    assert result.outcome is StopOutcome.COMPLETED
    assert result.text == "It's 5."
    assert result.messages == [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "5"}],
        },
        {"role": "assistant", "content": last.content},
    ]


async def test_run_token_usage_is_summed_across_iterations(fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[ADD_CALL], input_tokens=100, output_tokens=20),
        make_message("end_turn", text="5", input_tokens=130, output_tokens=8),
    )

    result = await run(client, "What is 2 + 3?", [make_add_tool()], model=MODEL)

    assert result.usage == TokenUsage(input_tokens=230, output_tokens=28)


async def test_run_unrecognised_stop_reason_after_tool_carries_summed_usage(
    fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[ADD_CALL], input_tokens=100, output_tokens=20),
        make_message("pause_turn", input_tokens=130, output_tokens=8),
    )

    with pytest.raises(UnexpectedStopReason) as exc_info:
        await run(client, "What is 2 + 3?", [make_add_tool()], model=MODEL)

    assert exc_info.value.usage == TokenUsage(input_tokens=230, output_tokens=28)
    assert len(exc_info.value.messages) == 4


async def test_run_exceeding_default_cap_of_5_raises_without_running_last_tools(
    fake_client, make_message
):
    client = fake_client(*[make_message("tool_use", tool_uses=[ADD_CALL]) for _ in range(6)])
    received: list[AddInput] = []

    with pytest.raises(MaxIterationsExceeded):
        await run(client, "Loop forever", [make_add_tool(received)], model=MODEL)

    assert len(client.messages.calls) == 5
    assert len(received) == 4


async def test_run_max_iterations_override_is_respected(fake_client, make_message):
    client = fake_client(*[make_message("tool_use", tool_uses=[ADD_CALL]) for _ in range(3)])

    with pytest.raises(MaxIterationsExceeded):
        await run(client, "Loop", [make_add_tool()], model=MODEL, max_iterations=2)

    assert len(client.messages.calls) == 2


async def test_run_max_iterations_error_carries_trace(fake_client, make_message):
    first = make_message("tool_use", tool_uses=[ADD_CALL], input_tokens=100, output_tokens=20)
    second = make_message("tool_use", tool_uses=[ADD_CALL], input_tokens=130, output_tokens=8)
    client = fake_client(first, second)

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await run(client, "Loop", [make_add_tool()], model=MODEL, max_iterations=2)

    error = exc_info.value
    assert error.messages == [
        {"role": "user", "content": "Loop"},
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "5"}],
        },
        {"role": "assistant", "content": second.content},
    ]
    assert error.usage == TokenUsage(input_tokens=230, output_tokens=28)


@pytest.mark.parametrize("max_iterations", [0, -1])
async def test_run_max_iterations_below_one_raises_before_calling_model(
    fake_client, max_iterations
):
    client = fake_client()

    with pytest.raises(ValueError, match="max_iterations"):
        await run(client, "Go", [], model=MODEL, max_iterations=max_iterations)

    assert client.messages.calls == []


async def test_run_duplicate_tool_names_raise_before_calling_model(fake_client):
    client = fake_client()

    with pytest.raises(ValueError, match="add"):
        await run(client, "Go", [make_add_tool(), make_add_tool()], model=MODEL)

    assert client.messages.calls == []


async def test_run_without_tools_omits_tools_param(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", [], model=MODEL)

    assert client.messages.calls[0]["tools"] is omit


async def test_run_tool_use_without_tool_blocks_raises_with_trace(fake_client, make_message):
    reply = make_message("tool_use", text="I'll use a tool.")
    client = fake_client(reply)

    with pytest.raises(UnexpectedStopReason) as exc_info:
        await run(client, "Go", [make_add_tool()], model=MODEL)

    assert exc_info.value.messages == [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": reply.content},
    ]
    assert len(client.messages.calls) == 1
