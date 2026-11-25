import asyncio

import pytest
from pydantic import BaseModel

from creative_ops.loop.agent_loop import MaxIterationsExceeded, run
from creative_ops.tools.tool import Tool

MODEL = "claude-haiku-5-5"  # the one model with a price entry


class Empty(BaseModel):
    pass


def noop_tool(handler=None) -> Tool:
    async def default(_: Empty) -> str:
        return "ok"

    return Tool(
        name="noop", description="Does nothing.", input_model=Empty, handler=handler or default
    )


async def test_run_is_an_agent_span_with_a_generation_per_iteration(
    fake_langfuse, fake_client, make_message
):
    client = fake_client(
        make_message(
            "tool_use",
            input_tokens=100,
            output_tokens=20,
            tool_uses=[{"id": "t1", "name": "noop", "input": {}}],
        ),
        make_message("end_turn", text="Done", input_tokens=200, output_tokens=10),
    )

    await run(client, "Go", [noop_tool()], model=MODEL, system="Be brief", agent="intake")

    [root] = fake_langfuse.roots
    assert root.name == "intake-run"
    assert root.fields["as_type"] == "agent"
    assert root.fields["output"] == "Done"
    assert root.fields["metadata"]["agent"] == "intake"
    assert root.fields["metadata"]["outcome"] == "completed"
    first, second = root.children
    assert [g.name for g in root.children] == ["call-model", "call-model"]
    assert first.fields["as_type"] == "generation"
    assert first.fields["model"] == MODEL
    assert first.fields["usage_details"] == {"input": 100, "output": 20}
    assert first.fields["cost_details"] == {"total": pytest.approx(0.00002)}
    assert first.fields["metadata"] == {"iteration": 1, "stop_reason": "tool_use", "thinking": None}
    assert first.fields["output"]["role"] == "assistant"
    assert first.fields["output"]["tool_calls"][0]["function"]["name"] == "noop"
    assert second.fields["input"][-1] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}
    assert first.fields["input"][0] == {"role": "system", "content": "Be brief"}
    assert second.fields["usage_details"] == {"input": 200, "output": 10}
    assert second.fields["metadata"]["stop_reason"] == "end_turn"


async def test_unpriced_model_has_tokens_and_no_cost(fake_langfuse, fake_client, make_message):
    await run(fake_client(make_message("end_turn", text="Hi")), "Go", [], model="test-model")

    [generation] = fake_langfuse.roots[0].children
    assert generation.fields["usage_details"] == {"input": 10, "output": 5}
    assert generation.fields["cost_details"] is None


async def test_failed_run_is_error_level_with_its_completed_iterations(
    fake_langfuse, fake_client, make_message
):
    tool_use = make_message("tool_use", tool_uses=[{"id": "t1", "name": "noop", "input": {}}])
    client = fake_client(tool_use, tool_use)

    with pytest.raises(MaxIterationsExceeded):
        await run(client, "Go", [noop_tool()], model=MODEL, max_iterations=2)

    [root] = fake_langfuse.roots
    assert root.fields["level"] == "ERROR"
    assert "MaxIterationsExceeded" in root.fields["status_message"]
    assert len(root.children) == 2


async def test_parallel_subagent_runs_nest_under_the_run_that_dispatched_them(
    fake_langfuse, routing_client, make_message
):
    async def dispatch(_: Empty) -> str:
        await asyncio.sleep(0)  # let the other call start first
        await run(client, "sub", [], model=MODEL, system="sub", agent="review")
        return "ok"

    client = routing_client(
        {
            "main": [
                make_message(
                    "tool_use",
                    tool_uses=[{"id": f"t{i}", "name": "noop", "input": {}} for i in (1, 2)],
                ),
                make_message("end_turn", text="Done"),
            ],
            "sub": [make_message("end_turn", text="a"), make_message("end_turn", text="b")],
        }
    )

    await run(client, "Go", [noop_tool(dispatch)], model=MODEL, system="main", agent="coordinator")

    [root] = fake_langfuse.roots
    assert root.name == "coordinator-run"
    reviews = [c for c in root.children if c.name == "review-run"]
    assert len(reviews) == 2
    assert all([g.name for g in r.children] == ["call-model"] for r in reviews)
    assert {r.fields["output"] for r in reviews} == {"a", "b"}


class BrokenLangfuse:
    """A client whose SDK fails: on open, or on every update."""

    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on

    def start_as_current_observation(self, **fields):
        if self.fail_on == "open":
            raise RuntimeError("langfuse down")
        return self._observation()

    def _observation(self):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield self

        return cm()

    def update(self, **fields):
        raise RuntimeError("langfuse down")


@pytest.mark.parametrize("fail_on", ["open", "update"])
async def test_a_failing_recorder_does_not_fail_the_run(
    monkeypatch, fake_client, make_message, fail_on
):
    monkeypatch.setattr("creative_ops.tracing._get_client", lambda: BrokenLangfuse(fail_on))

    result = await run(fake_client(make_message("end_turn", text="Hi")), "Go", [], model=MODEL)

    assert result.text == "Hi"
