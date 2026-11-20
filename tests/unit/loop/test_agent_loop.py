import json
import os
import subprocess
import sys

import pytest
from anthropic import omit

from creative_ops.config import get_settings
from creative_ops.loop.agent_loop import StopOutcome, TokenUsage, UnexpectedStopReason, run

MODEL = "test-model"


async def test_run_end_turn_returns_completed_with_text(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="Hello!"))

    result = await run(client, "Say hello", model=MODEL, max_tokens=256)

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

    result = await run(client, "Go", model=MODEL)

    assert result.outcome is expected


async def test_run_unrecognised_stop_reason_raises(fake_client, make_message):
    client = fake_client(make_message("pause_turn"))

    with pytest.raises(UnexpectedStopReason, match="pause_turn"):
        await run(client, "Go", model=MODEL)


async def test_run_unrecognised_stop_reason_error_carries_trace(fake_client, make_message):
    reply = make_message("pause_turn", text="thinking", input_tokens=12, output_tokens=7)
    client = fake_client(reply)

    with pytest.raises(UnexpectedStopReason) as exc_info:
        await run(client, "Go", model=MODEL)

    error = exc_info.value
    assert error.stop_reason == "pause_turn"
    assert error.messages == [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": reply.content},
    ]
    assert error.usage == TokenUsage(input_tokens=12, output_tokens=7)


async def test_run_with_system_sends_system_prompt(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", model=MODEL, system="You are terse.")

    assert client.messages.calls[0]["system"] == "You are terse."


async def test_run_without_system_omits_system_param(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok"))

    await run(client, "Go", model=MODEL)

    assert client.messages.calls[0]["system"] is omit


async def test_run_completed_returns_full_message_history(fake_client, make_message):
    reply = make_message("end_turn", text="Hi there")
    client = fake_client(reply)

    result = await run(client, "Say hi", model=MODEL)

    assert result.messages == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": reply.content},
    ]


async def test_run_completed_returns_token_usage(fake_client, make_message):
    client = fake_client(make_message("end_turn", text="ok", input_tokens=12, output_tokens=7))

    result = await run(client, "Go", model=MODEL)

    assert result.usage == TokenUsage(input_tokens=12, output_tokens=7)


async def test_run_without_model_uses_configured_fast_model(fake_client, make_message, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("MODEL_FAST", "fast-test-model")
    get_settings.cache_clear()
    client = fake_client(make_message("end_turn", text="ok"))

    try:
        await run(client, "Go")
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

    result = await run(client, "Say hi", model=MODEL)

    dumped = json.loads(result.model_dump_json())
    assert dumped["messages"][1]["content"][0]["text"] == "Hi there"
