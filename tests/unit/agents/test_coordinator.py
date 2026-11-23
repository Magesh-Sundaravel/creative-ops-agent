import json
import shutil
from datetime import date

import pytest
from mcp import Client

from creative_ops.agents import coordinator, handoff, intake, review
from creative_ops.agents.coordinator import coordinate
from creative_ops.loop.agent_loop import MaxIterationsExceeded
from mcp_server.server import SEED_DIR, build_server

TODAY = date(2030, 1, 15)
BRIEF = (
    "Loomhouse, Spring Linen Edit: launch the linen range with an Instagram image, "
    "due 1 March 2031."
)
CAMPAIGN = {
    "brand": "Loomhouse",
    "campaign_name": "Spring Linen Edit",
    "objective": "Launch the linen range",
    "deliverables": [{"asset_type": "image", "channel": "Instagram"}],
    "deadline": "2031-03-01",
}
KEY = {"brand": "Loomhouse", "campaign_name": "Spring Linen Edit"}


def route(module) -> str:
    """The fixed opening of an agent's system prompt, before any placeholder."""
    return module.SYSTEM_PROMPT.split("{")[0]


COORDINATOR, INTAKE, REVIEW, HANDOFF = (route(m) for m in (coordinator, intake, review, handoff))


@pytest.fixture
def connect_without_flaws(tmp_path):
    """Like `connect`, but the seed has no Spring Linen Edit Asset that fails a check."""
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    assets = json.loads((tmp_path / "assets.json").read_text())
    (tmp_path / "assets.json").write_text(json.dumps([a for a in assets if a["id"] != "AST-009"]))
    return lambda: Client(build_server(tmp_path))


def call(tool_id: str, name: str, tool_input: dict | None = None) -> dict:
    return {"id": tool_id, "name": name, "input": tool_input or {}}


@pytest.fixture
def clean_run(routing_client, make_message):
    """A client scripted for one clean campaign: intake, review, handoff, then done.

    `clean_run(*calls)` scripts the Coordinator with those calls, one per response, instead.
    Intake can run twice; each Run submits the same campaign.
    """
    default = [call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)]
    return lambda *calls: routing_client(
        {
            COORDINATOR: [
                *(make_message("tool_use", tool_uses=[c]) for c in calls or default),
                make_message("end_turn", text="All done."),
            ],
            INTAKE: 2
            * [
                make_message(
                    "tool_use",
                    tool_uses=[call("i1", "submit_brief", {"briefs": [CAMPAIGN]})],
                ),
                make_message("end_turn", text="intake"),
            ],
            REVIEW: [
                make_message(
                    "tool_use",
                    tool_uses=[
                        call(
                            "r1",
                            "submit_review",
                            {"findings": [], "coverage": {"0": ["AST-001"]}},
                        )
                    ],
                ),
                make_message("end_turn", text="review"),
            ],
            HANDOFF: [
                make_message(
                    "tool_use",
                    tool_uses=[
                        call(
                            "h1",
                            "tracker_create_task",
                            {"title": "Spring Linen Edit", "description": "IG image."},
                        ),
                        call("h2", "submit_handoff", {"production_notes": "Lead with AST-001."}),
                    ],
                ),
                make_message("end_turn", text="handoff"),
            ],
        }
    )


class RecordCalls:
    """A session that records the name of every tool call it passes to the server."""

    def __init__(self, session) -> None:
        self._session = session
        self.names: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict):
        self.names.append(name)
        return await self._session.call_tool(name, arguments)


async def test_coordinate_clean_campaign_returns_handoff_package_and_creates_one_task(
    connect_without_flaws, clean_run
):
    client = clean_run()

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    assert [(p.brand, p.campaign_name) for p in result.handoffs] == [
        ("loomhouse", "Spring Linen Edit")
    ]
    assert result.handoffs[0].production_notes == "Lead with AST-001."
    assert recorded.names.count("tracker_create_task") == 1


async def test_coordinate_result_carries_coordinator_and_every_subagent_loop_result(
    connect_without_flaws, clean_run
):
    client = clean_run()

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert result.loop_result.text == "All done."
    assert [i.loop_result.text for i in result.intakes] == ["intake"]
    assert [r.loop_result.text for r in result.reviews] == ["review"]
    assert [h.loop_result.text for h in result.handoffs] == ["handoff"]


async def test_coordinate_offers_model_only_intake_review_and_handoff(
    connect_without_flaws, clean_run
):
    client = clean_run()

    async with connect_without_flaws() as session:
        await coordinate(client, session, BRIEF, today=TODAY)

    offered = [
        sorted(tool["name"] for tool in c["tools"]) for c in client.messages.calls[COORDINATOR]
    ]
    assert offered == [["handoff", "intake", "review"]] * 4


async def test_coordinate_hitting_its_iteration_cap_raises(
    connect_without_flaws, routing_client, make_message
):
    # Asks for an unknown campaign's review forever; each call gets an is_error result.
    stuck = make_message("tool_use", tool_uses=[call("c1", "review", KEY)])
    client = routing_client({COORDINATOR: [stuck] * 8})

    async with connect_without_flaws() as session:
        with pytest.raises(MaxIterationsExceeded) as raised:
            await coordinate(client, session, BRIEF, today=TODAY)

    assert raised.value.max_iterations == 8


async def test_coordinate_handoff_after_a_new_intake_is_refused_until_reviewed_again(
    connect_without_flaws, clean_run
):
    client = clean_run(
        call("c1", "intake"),
        call("c2", "review", KEY),
        call("c3", "intake"),
        call("c4", "handoff", KEY),
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    handoff_result = client.messages.calls[COORDINATOR][4]["messages"][-1]["content"][0]
    assert handoff_result["is_error"] is True
    assert result.handoffs == []
    assert "tracker_create_task" not in recorded.names
