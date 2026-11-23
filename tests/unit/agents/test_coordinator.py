import json
import shutil
from collections.abc import Sequence
from datetime import date

import pytest
from mcp import Client

from creative_ops.agents import coordinator, handoff, intake, review
from creative_ops.agents.coordinator import Amend, CoordinatorState, coordinate, resume
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
def script(make_message):
    """Scripted responses per agent, to build a routing client from."""

    class Script:
        @staticmethod
        def coordinator(*calls: dict) -> list:
            """One response per call, then end_turn."""
            return [
                *(make_message("tool_use", tool_uses=[c]) for c in calls),
                make_message("end_turn", text="All done."),
            ]

        @staticmethod
        def intake(*briefs: dict, ambiguities: Sequence[str] = ()) -> list:
            submission = {"briefs": list(briefs), "ambiguities": list(ambiguities)}
            return [
                make_message("tool_use", tool_uses=[call("i1", "submit_brief", submission)]),
                make_message("end_turn", text="intake"),
            ]

        @staticmethod
        def review() -> list:
            return [
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
            ]

        @staticmethod
        def handoff() -> list:
            return [
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
            ]

    return Script


@pytest.fixture
def clean_run(routing_client, script):
    """A client scripted for one clean campaign: intake, review, handoff, then done.

    `clean_run(*calls)` scripts the Coordinator with those calls, one per response, instead.
    """
    default = [call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)]
    return lambda *calls: routing_client(
        {
            COORDINATOR: script.coordinator(*(calls or default)),
            INTAKE: script.intake(CAMPAIGN),
            REVIEW: script.review(),
            HANDOFF: script.handoff(),
        }
    )


class RecordCalls:
    """A session that records every tool call name and resource URI it passes to the server."""

    def __init__(self, session) -> None:
        self._session = session
        self.names: list[str] = []
        self.uris: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict):
        self.names.append(name)
        return await self._session.call_tool(name, arguments)

    async def read_resource(self, uri: str):
        self.uris.append(uri)
        return await self._session.read_resource(uri)


async def test_coordinate_clean_campaign_returns_handoff_package_and_creates_one_task(
    connect_without_flaws, clean_run
):
    client = clean_run()

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.escalation is None
    assert (outcome.handoff.brand, outcome.handoff.campaign_name) == (
        "loomhouse",
        "Spring Linen Edit",
    )
    assert outcome.handoff.production_notes == "Lead with AST-001."
    assert result.escalation is None
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
    assert [c.handoff.loop_result.text for c in result.campaigns] == ["handoff"]


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


async def test_coordinate_second_intake_in_a_run_is_refused(connect_without_flaws, clean_run):
    client = clean_run(call("c1", "intake"), call("c2", "intake"))

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    second = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert second["is_error"] is True
    assert len(result.intakes) == 1


async def test_coordinate_intake_can_run_again_after_its_run_fails(
    connect_without_flaws, routing_client, make_message
):
    client = routing_client(
        {
            COORDINATOR: [
                make_message("tool_use", tool_uses=[call("c1", "intake")]),
                make_message("tool_use", tool_uses=[call("c2", "intake")]),
                make_message("end_turn"),
            ],
            INTAKE: [
                make_message("end_turn", text="No submission."),
                make_message(
                    "tool_use", tool_uses=[call("i1", "submit_brief", {"briefs": [CAMPAIGN]})]
                ),
                make_message("end_turn"),
            ],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    second = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert "is_error" not in second
    assert len(result.intakes) == 1


async def test_coordinate_two_intakes_in_one_response_run_intake_once(
    connect_without_flaws, routing_client, make_message
):
    client = routing_client(
        {
            COORDINATOR: [
                make_message("tool_use", tool_uses=[call("c1", "intake"), call("c2", "intake")]),
                make_message("end_turn"),
            ],
            INTAKE: 2
            * [
                make_message(
                    "tool_use", tool_uses=[call("i1", "submit_brief", {"briefs": [CAMPAIGN]})]
                ),
                make_message("end_turn"),
            ],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert len(client.messages.calls[INTAKE]) == 2  # one Intake Run: submit, then end_turn
    assert len(result.intakes) == 1


async def test_coordinate_handoff_before_review_is_refused_and_creates_no_task(
    connect_without_flaws, clean_run
):
    client = clean_run(call("c1", "intake"), call("c2", "handoff", KEY))

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    refused = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert refused["is_error"] is True
    assert result.campaigns[0].handoff is None
    assert "tracker_create_task" not in recorded.names


async def test_coordinate_missing_critical_field_escalates_campaign_and_never_reviews_it(
    connect_without_flaws, routing_client, script
):
    no_deadline = {**CAMPAIGN, "deadline": None}
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake")),
            INTAKE: script.intake(no_deadline, ambiguities=["Budget unclear"]),
            REVIEW: [],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.handoff is None
    assert outcome.escalation.reason == "missing_critical_field"
    assert outcome.escalation.answer == "amend"
    assert outcome.escalation.needs == ["deadline"]
    assert outcome.escalation.ambiguities == ["Budget unclear"]
    assert (outcome.brand, outcome.campaign_name) == ("Loomhouse", "Spring Linen Edit")
    assert client.messages.calls[REVIEW] == []


async def test_coordinate_review_of_campaign_with_critical_missing_is_permission_denied(
    connect_without_flaws, routing_client, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake"), call("c2", "review", KEY)),
            INTAKE: script.intake({**CAMPAIGN, "deadline": None}),
            REVIEW: [],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    denied = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert denied["is_error"] is True
    assert json.loads(denied["content"])["type"] == "permission_denied"
    assert client.messages.calls[REVIEW] == []
    assert result.campaigns[0].escalation.reason == "missing_critical_field"


async def test_coordinate_cannot_review_escalates_campaign_without_retry(
    connect_without_flaws, routing_client, script
):
    unknown = {"brand": "Nobody", "campaign_name": "Spring Linen Edit"}
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", unknown), call("c3", "review", unknown)
            ),
            INTAKE: script.intake({**CAMPAIGN, **unknown}, ambiguities=["Tone unclear"]),
            REVIEW: [],
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.escalation.reason == "cannot_review"
    assert outcome.escalation.answer == "amend"
    assert outcome.escalation.needs == ["Brand 'Nobody' is not a known client brand"]
    assert outcome.escalation.ambiguities == ["Tone unclear"]
    for i in (2, 3):  # the review that escalated, then the retry
        refused = client.messages.calls[COORDINATOR][i]["messages"][-1]["content"][0]
        assert refused["is_error"] is True
        assert json.loads(refused["content"])["type"] == "permission_denied"
    assert recorded.uris.count("brand://brands") == 1  # Review was attempted once, not retried
    assert result.reviews == []


async def test_coordinate_zero_structured_briefs_escalates_the_whole_brief(
    connect_without_flaws, routing_client, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake")),
            INTAKE: script.intake(ambiguities=["Text is not a Brief"]),
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, "What's the weather?", today=TODAY)

    assert result.campaigns == []
    assert result.escalation.reason == "not_a_brief"
    assert result.escalation.answer == "amend"
    assert result.escalation.needs
    assert result.escalation.ambiguities == ["Text is not a Brief"]


async def test_coordinate_campaign_the_model_never_finishes_is_listed_as_unfinished(
    connect_without_flaws, clean_run
):
    client = clean_run(call("c1", "intake"))

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert (outcome.brand, outcome.campaign_name) == ("Loomhouse", "Spring Linen Edit")
    assert outcome.handoff is None
    assert outcome.escalation is None


async def test_coordinate_state_returned_with_an_escalation_round_trips_through_json(
    connect_without_flaws, routing_client, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake")),
            INTAKE: script.intake({**CAMPAIGN, "deadline": None}, ambiguities=["Budget unclear"]),
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert result.state.brief == BRIEF
    assert result.state.escalations == [result.campaigns[0].escalation]
    assert CoordinatorState.model_validate_json(result.state.model_dump_json()) == result.state


async def test_coordinate_without_an_escalation_returns_no_state(connect_without_flaws, clean_run):
    client = clean_run()

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert result.state is None


async def test_resume_with_amend_reruns_intake_on_amended_brief_and_reaches_handoff(
    connect_without_flaws, routing_client, script
):
    full = [call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)]
    client = routing_client(
        {
            COORDINATOR: [*script.coordinator(call("c1", "intake")), *script.coordinator(*full)],
            INTAKE: [*script.intake({**CAMPAIGN, "deadline": None}), *script.intake(CAMPAIGN)],
            REVIEW: script.review(),
            HANDOFF: script.handoff(),
        }
    )
    brief = "Loomhouse, Spring Linen Edit: launch the linen range with an Instagram image."

    async with connect_without_flaws() as session:
        escalated = await coordinate(client, session, brief, today=TODAY)
        state = CoordinatorState.model_validate_json(escalated.state.model_dump_json())
        result = await resume(client, session, state, Amend(text="Due 1 March 2031."), TODAY)

    amended_prompt = client.messages.calls[INTAKE][2]["messages"][0]["content"]
    assert brief in amended_prompt
    assert "Due 1 March 2031." in amended_prompt
    [outcome] = result.campaigns
    assert outcome.escalation is None
    assert outcome.handoff.deadline == date(2031, 3, 1)
    assert result.state is None
