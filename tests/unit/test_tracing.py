from types import SimpleNamespace

import pytest

from creative_ops import config, tracing
from creative_ops.tracing import _get_client as real_get_client  # the autouse fixture patches it


@pytest.fixture
def fresh_client(monkeypatch):
    """Settings and the cached client rebuilt from the test's environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_built", False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_no_keys_means_tracing_is_off(fresh_client):
    assert real_get_client() is None


def test_only_one_key_means_tracing_is_off(fresh_client, monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")

    assert real_get_client() is None


def test_both_keys_mean_tracing_is_on(fresh_client, monkeypatch):
    built = []
    monkeypatch.setattr(tracing, "Langfuse", lambda **kwargs: built.append(kwargs) or "client")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-1")
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    assert real_get_client() == "client"
    assert built == [
        {
            "public_key": "pk-lf-1",
            "secret_key": "sk-lf-1",
            "base_url": "https://us.cloud.langfuse.com",
            "should_export_span": tracing._should_export,
        }
    ]


def _span(scope_name: str | None):
    scope = None if scope_name is None else SimpleNamespace(name=scope_name)
    return SimpleNamespace(instrumentation_scope=scope, attributes={})


def test_should_export_anthropic_sdk_span_drops_it():
    assert tracing._should_export(_span(tracing.ANTHROPIC_SDK_SCOPE)) is False


def test_should_export_langfuse_span_keeps_it():
    assert tracing._should_export(_span("langfuse-sdk")) is True


def test_chat_tool_use_and_result_pair_by_call_id():
    messages = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Calling"},
                {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "2"}]},
    ]

    assert tracing.chat(messages) == [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": "Calling",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "2"},
    ]


def test_thinking_joins_thinking_blocks_and_drops_the_signature():
    content = [
        {"type": "thinking", "thinking": "first", "signature": "sig"},
        {"type": "text", "text": "answer"},
        {"type": "thinking", "thinking": "second", "signature": "sig"},
    ]

    assert tracing.thinking(content) == "first\n\nsecond"
    assert tracing.assistant_message(content) == {"role": "assistant", "content": "answer"}
