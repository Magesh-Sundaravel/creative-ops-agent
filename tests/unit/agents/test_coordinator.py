import json
import shutil
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from creative_ops.agents import coordinator, handoff, intake, review
from creative_ops.agents.coordinator import (
    Acknowledge,
    Amend,
    AnswerDoesNotFit,
    CoordinatorState,
    Retry,
    coordinate,
    resume,
)
from creative_ops.loop.agent_loop import MaxIterationsExceeded
from creative_ops.schemas.client_stack import permission_denied
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
FLAWED = {"AST-009", "AST-014", "AST-015"}


def route(module) -> str:
    """The fixed opening of an agent's system prompt, before any placeholder."""
    return module.SYSTEM_PROMPT.split("{")[0]


COORDINATOR, INTAKE, REVIEW, HANDOFF = (route(m) for m in (coordinator, intake, review, handoff))


@pytest.fixture
def connect_without_flaws(tmp_path):
    """Like `connect`, but no Spring Linen Edit or City Miles Asset fails a check."""
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    assets = json.loads((tmp_path / "assets.json").read_text())
    (tmp_path / "assets.json").write_text(json.dumps([a for a in assets if a["id"] not in FLAWED]))
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
        def review(*findings: dict) -> list:
            submission = {"findings": list(findings), "coverage": {"0": ["AST-001"]}}
            return [
                make_message("tool_use", tool_uses=[call("r1", "submit_review", submission)]),
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


async def test_coordinate_intake_failing_once_is_retried_without_an_escalation(
    connect_without_flaws, routing_client, make_message, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake")),
            INTAKE: [make_message("end_turn", text="No submission."), *script.intake(CAMPAIGN)],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    first = client.messages.calls[COORDINATOR][1]["messages"][-1]["content"][0]
    assert "is_error" not in first
    assert len(result.intakes) == 1
    assert result.escalation is None
    assert [f.error for f in result.failed_runs] == ["NoValidSubmission"]


async def test_coordinate_intake_failing_twice_escalates_the_whole_brief_accepting_retry(
    connect_without_flaws, routing_client, make_message, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake"), call("c2", "intake")),
            INTAKE: [make_message("end_turn", text="No submission.")] * 2,
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert result.escalation.reason == "run_failed"
    assert result.escalation.answer == "retry"
    assert result.escalation.needs == ["Intake failed twice (NoValidSubmission)"]
    assert result.escalation.ambiguities == []
    assert result.campaigns == []
    for i in (1, 2):  # the intake that escalated, then the model asking again
        refused = client.messages.calls[COORDINATOR][i]["messages"][-1]["content"][0]
        assert refused["is_error"] is True
    assert len(client.messages.calls[INTAKE]) == 2
    assert result.state.escalations == [result.escalation]


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


CAMPAIGN_2 = {
    "brand": "Stridewell",
    "campaign_name": "City Miles",
    "objective": "Launch the running range",
    "deliverables": [{"asset_type": "video", "channel": "TikTok"}],
    "deadline": "2030-06-01",
}
KEY_2 = {"brand": "Stridewell", "campaign_name": "City Miles"}
TWO_CAMPAIGNS = f"{BRIEF} Stridewell, City Miles: a TikTok video, due 1 June 2030."


class CampaignRoutingMessages:
    """Like the routing client, but a `(prefix, campaign_name)` route also needs that campaign
    name in the Run's first message, so concurrent Runs of the same agent stay deterministic.
    A route with campaign_name `None` matches any Run of that agent.
    """

    def __init__(self, routes: dict[tuple[str, str | None], list]) -> None:
        self._routes = {key: list(responses) for key, responses in routes.items()}
        self.calls: dict[tuple[str, str | None], list[dict]] = {key: [] for key in routes}

    async def create(self, **kwargs):
        first = kwargs["messages"][0]["content"]
        for (prefix, name), responses in self._routes.items():
            if kwargs["system"].startswith(prefix) and (name is None or name in first):
                self.calls[(prefix, name)].append(kwargs)
                if not responses:
                    raise AssertionError(f"Route {(prefix, name)!r} has no responses left")
                return responses.pop(0)
        raise AssertionError(f"No route for system prompt {kwargs['system']!r}")


class CampaignRoutingClient:
    def __init__(self, routes: dict[tuple[str, str | None], list]) -> None:
        self.messages = CampaignRoutingMessages(routes)


def review_of(make_message, asset_id: str) -> list:
    """A Review Run that covers deliverable 0 with one Asset and finds nothing."""
    submission = {"findings": [], "coverage": {"0": [asset_id]}}
    return [
        make_message("tool_use", tool_uses=[call("r1", "submit_review", submission)]),
        make_message("end_turn", text="review"),
    ]


def coordinator_turns(make_message, *turns: list[dict]) -> list:
    """One Coordinator response per turn, each issuing all that turn's calls, then end_turn."""
    return [
        *(make_message("tool_use", tool_uses=t) for t in turns),
        make_message("end_turn", text="All done."),
    ]


@pytest.fixture
def two_campaigns(make_message, script):
    """A client for a two-campaign Brief: `two_campaigns(*turns, intake=..., second=...)`."""

    def build(*turns: list[dict], intake: list | None = None) -> CampaignRoutingClient:
        return CampaignRoutingClient(
            {
                (COORDINATOR, None): coordinator_turns(make_message, *turns),
                (INTAKE, None): intake or script.intake(CAMPAIGN, CAMPAIGN_2),
                (REVIEW, "Spring Linen Edit"): review_of(make_message, "AST-001"),
                (REVIEW, "City Miles"): review_of(make_message, "AST-002"),
                (HANDOFF, "Spring Linen Edit"): script.handoff(),
                (HANDOFF, "City Miles"): script.handoff(),
            }
        )

    return build


def results_of(client, turn: int) -> list[dict]:
    """The tool results the Coordinator saw after its `turn`-th response (1-based)."""
    return client.messages.calls[(COORDINATOR, None)][turn]["messages"][-1]["content"]


async def test_coordinate_two_campaigns_reviewed_in_one_response_both_reach_handoff(
    connect_without_flaws, two_campaigns
):
    client = two_campaigns(
        [call("c1", "intake")],
        [call("c2", "review", KEY), call("c3", "review", KEY_2)],
        [call("c4", "handoff", KEY), call("c5", "handoff", KEY_2)],
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, TWO_CAMPAIGNS, today=TODAY)

    assert [(c.campaign_name, c.escalation) for c in result.campaigns] == [
        ("Spring Linen Edit", None),
        ("City Miles", None),
    ]
    assert all(c.handoff is not None for c in result.campaigns)
    assert all("is_error" not in r for r in results_of(client, 2))
    assert recorded.names.count("tracker_create_task") == 2


async def test_coordinate_one_campaign_escalating_leaves_the_other_to_reach_handoff(
    connect_without_flaws, two_campaigns, script
):
    unknown = {**KEY_2, "brand": "Nobody"}
    client = two_campaigns(
        [call("c1", "intake")],
        [call("c2", "review", KEY), call("c3", "review", unknown)],
        [call("c4", "handoff", KEY)],
        intake=script.intake(CAMPAIGN, {**CAMPAIGN_2, **unknown}),
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, TWO_CAMPAIGNS, today=TODAY)

    linen, miles = result.campaigns
    assert linen.handoff is not None
    assert linen.escalation is None
    assert miles.handoff is None
    assert miles.escalation.reason == "cannot_review"
    assert result.state.escalations == [miles.escalation]


async def test_coordinate_two_handoffs_of_one_campaign_in_one_response_create_one_task(
    connect_without_flaws, two_campaigns
):
    client = two_campaigns(
        [call("c1", "intake")],
        [call("c2", "review", KEY)],
        [call("c3", "handoff", KEY), call("c4", "handoff", KEY)],
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, TWO_CAMPAIGNS, today=TODAY)

    first, second = results_of(client, 3)
    assert "is_error" not in first
    assert second["is_error"] is True
    assert json.loads(second["content"])["type"] == "permission_denied"
    assert recorded.names.count("tracker_create_task") == 1
    assert result.campaigns[0].handoff is not None


async def test_resume_skips_a_campaign_already_handed_off_and_creates_no_duplicate_task(
    connect_without_flaws, make_message, script
):
    no_deadline = {**CAMPAIGN_2, "deadline": None}
    both = [call("c2", "review", KEY), call("c3", "review", KEY_2)]
    client = CampaignRoutingClient(
        {
            (COORDINATOR, None): [
                *coordinator_turns(
                    make_message,
                    [call("c1", "intake")],
                    [call("c2", "review", KEY)],
                    [call("c3", "handoff", KEY)],
                ),
                *coordinator_turns(
                    make_message,
                    [call("c1", "intake")],
                    both,
                    [call("c4", "handoff", KEY), call("c5", "handoff", KEY_2)],
                ),
            ],
            (INTAKE, None): [
                *script.intake(CAMPAIGN, no_deadline),
                *script.intake(CAMPAIGN, CAMPAIGN_2),
            ],
            (REVIEW, "Spring Linen Edit"): review_of(make_message, "AST-001"),
            (REVIEW, "City Miles"): review_of(make_message, "AST-002"),
            (HANDOFF, "Spring Linen Edit"): script.handoff(),
            (HANDOFF, "City Miles"): script.handoff(),
        }
    )
    brief = f"{BRIEF} Stridewell, City Miles: a TikTok video."

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        escalated = await coordinate(client, recorded, brief, today=TODAY)
        state = CoordinatorState.model_validate_json(escalated.state.model_dump_json())
        result = await resume(client, recorded, state, Amend(text="Due 1 June 2030."), TODAY)

    calls = client.messages.calls[(COORDINATOR, None)]
    resumed_intake = json.loads(calls[5]["messages"][-1]["content"][0]["content"])
    assert [c["handed_off"] for c in resumed_intake["campaigns"]] == [True, False]
    for denied in (calls[6]["messages"][-1]["content"][0], calls[7]["messages"][-1]["content"][0]):
        assert denied["is_error"] is True
        assert json.loads(denied["content"])["type"] == "permission_denied"
    assert recorded.names.count("tracker_create_task") == 2
    linen, miles = result.campaigns
    assert linen.handoff.model_dump_json() == escalated.campaigns[0].handoff.model_dump_json()
    assert miles.handoff is not None
    assert result.state is None


async def test_campaign_with_no_name_escalates_and_again_after_an_amend(
    connect_without_flaws, routing_client, script
):
    unnamed = {**CAMPAIGN, "campaign_name": None}
    client = routing_client(
        {
            COORDINATOR: 2 * script.coordinator(call("c1", "intake")),
            INTAKE: 2 * script.intake(unnamed),
            REVIEW: [],
        }
    )

    async with connect_without_flaws() as session:
        escalated = await coordinate(client, session, BRIEF, today=TODAY)
        result = await resume(client, session, escalated.state, Amend(text="Budget £5k."), TODAY)

    for run in (escalated, result):
        [outcome] = run.campaigns
        assert (outcome.brand, outcome.campaign_name) == ("Loomhouse", None)
        assert outcome.escalation.reason == "unnamed_campaign"
        assert outcome.escalation.answer == "amend"
        assert outcome.escalation.needs == ["campaign_name"]


async def test_coordinate_two_campaigns_sharing_a_key_both_escalate_asking_for_an_amend(
    connect_without_flaws, routing_client, script
):
    instagram, website = (
        {**CAMPAIGN, "deliverables": [{"asset_type": "image", "channel": channel}]}
        for channel in ("Instagram", "website")
    )
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake"), call("c2", "review", KEY)),
            INTAKE: script.intake(instagram, website),
            REVIEW: [],
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    assert len(result.campaigns) == 2
    for outcome in result.campaigns:
        assert outcome.escalation.reason == "duplicate_campaign"
        assert outcome.escalation.answer == "amend"
    assert result.state.escalations == [c.escalation for c in result.campaigns]
    denied = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert json.loads(denied["content"])["type"] == "permission_denied"
    assert client.messages.calls[REVIEW] == []


async def test_coordinate_refuses_review_and_handoff_of_a_campaign_while_it_is_being_reviewed(
    connect_without_flaws, make_message, script
):
    client = CampaignRoutingClient(
        {
            (COORDINATOR, None): coordinator_turns(
                make_message,
                [call("c1", "intake")],
                [call("c2", "review", KEY)],
                [call("c3", "review", KEY), call("c4", "review", KEY), call("c5", "handoff", KEY)],
            ),
            (INTAKE, None): script.intake(CAMPAIGN),
            (REVIEW, None): 2 * review_of(make_message, "AST-001"),
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    reviewed, again, handed = results_of(client, 3)
    assert "is_error" not in reviewed
    for refused in (again, handed):
        assert refused["is_error"] is True
        assert json.loads(refused["content"])["type"] == "permission_denied"
    assert len(result.reviews) == 2
    assert "tracker_create_task" not in recorded.names


BLOCKING = {
    "subject": "AST-001",
    "rule": "tone",
    "explanation": "Too dark for a spring launch.",
    "severity": "blocking",
}


@pytest.fixture
def rejected_run(routing_client, script):
    """A client for one campaign whose review requests revisions; then the model hands it off."""
    return lambda: routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN, ambiguities=["Budget unclear"]),
            REVIEW: script.review(BLOCKING),
            HANDOFF: [],
        }
    )


async def test_coordinate_revisions_requested_escalates_campaign_with_blocking_findings(
    connect_without_flaws, rejected_run
):
    async with connect_without_flaws() as session:
        result = await coordinate(client := rejected_run(), session, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.handoff is None
    assert outcome.escalation.reason == "revisions_requested"
    assert outcome.escalation.answer == "acknowledge"
    assert outcome.escalation.needs == ["AST-001 (tone): Too dark for a spring launch."]
    assert outcome.escalation.ambiguities == ["Budget unclear"]
    assert result.state.escalations == [outcome.escalation]
    assert len(client.messages.calls[REVIEW]) == 2  # one Review Run, not retried


async def test_coordinate_handoff_of_campaign_not_approved_is_permission_denied_and_creates_no_task(
    connect_without_flaws, rejected_run
):
    client = rejected_run()

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        await coordinate(client, recorded, BRIEF, today=TODAY)

    denied = client.messages.calls[COORDINATOR][3]["messages"][-1]["content"][0]
    assert denied["is_error"] is True
    assert json.loads(denied["content"])["type"] == "permission_denied"
    assert client.messages.calls[HANDOFF] == []
    assert "tracker_create_task" not in recorded.names


async def test_resume_with_acknowledge_closes_campaign_as_revisions_requested_without_a_run(
    connect_without_flaws, rejected_run
):
    client = rejected_run()

    async with connect_without_flaws() as session:
        escalated = await coordinate(client, session, BRIEF, today=TODAY)
        calls = {route: len(made) for route, made in client.messages.calls.items()}
        state = CoordinatorState.model_validate_json(escalated.state.model_dump_json())
        result = await resume(client, session, state, Acknowledge(), TODAY)

    assert {route: len(made) for route, made in client.messages.calls.items()} == calls
    [outcome] = result.campaigns
    assert outcome.revisions_requested is True
    assert outcome.escalation is None
    assert outcome.handoff is None
    assert result.loop_result is None
    assert result.state is None


@pytest.mark.parametrize(
    ("brief_campaign", "review_findings", "answer"),
    [
        (CAMPAIGN, [BLOCKING], Amend(text="Due 1 March 2031.")),
        ({**CAMPAIGN, "deadline": None}, [], Acknowledge()),
        ({**CAMPAIGN, "deadline": None}, [], Retry()),
    ],
    ids=["amend-on-revisions-requested", "acknowledge-on-missing-field", "retry-on-missing-field"],
)
async def test_resume_with_an_answer_that_does_not_fit_the_escalation_is_rejected(
    connect_without_flaws, routing_client, script, brief_campaign, review_findings, answer
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(call("c1", "intake"), call("c2", "review", KEY)),
            INTAKE: script.intake(brief_campaign),
            REVIEW: script.review(*review_findings),
        }
    )

    async with connect_without_flaws() as session:
        escalated = await coordinate(client, session, BRIEF, today=TODAY)
        with pytest.raises(AnswerDoesNotFit):
            await resume(client, session, escalated.state, answer, TODAY)


async def test_resume_with_amend_never_reviews_a_revisions_requested_campaign_again(
    connect_without_flaws, make_message, script
):
    no_deadline = {**CAMPAIGN_2, "deadline": None}
    both = [call("c2", "review", KEY), call("c3", "review", KEY_2)]
    client = CampaignRoutingClient(
        {
            (COORDINATOR, None): [
                *coordinator_turns(make_message, [call("c1", "intake")], [both[0]]),
                *coordinator_turns(
                    make_message,
                    [call("c1", "intake")],
                    both,
                    [call("c4", "handoff", KEY), call("c5", "handoff", KEY_2)],
                ),
            ],
            (INTAKE, None): [
                *script.intake(CAMPAIGN, no_deadline),
                *script.intake(CAMPAIGN, CAMPAIGN_2),
            ],
            (REVIEW, "Spring Linen Edit"): script.review(BLOCKING),
            (REVIEW, "City Miles"): review_of(make_message, "AST-002"),
            (HANDOFF, "City Miles"): script.handoff(),
        }
    )
    brief = f"{BRIEF} Stridewell, City Miles: a TikTok video."

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        escalated = await coordinate(client, recorded, brief, today=TODAY)
        state = CoordinatorState.model_validate_json(escalated.state.model_dump_json())
        result = await resume(client, recorded, state, Amend(text="Due 1 June 2030."), TODAY)

    calls = client.messages.calls[(COORDINATOR, None)]
    for denied in (calls[5]["messages"][-1]["content"][0], calls[6]["messages"][-1]["content"][0]):
        assert denied["is_error"] is True
        assert json.loads(denied["content"])["type"] == "permission_denied"
    assert len(client.messages.calls[(REVIEW, "Spring Linen Edit")]) == 2  # first Run only
    assert recorded.names.count("tracker_create_task") == 1
    linen, miles = result.campaigns
    assert linen.escalation == escalated.campaigns[0].escalation
    assert miles.handoff is not None
    assert result.state.escalations == [linen.escalation]


async def test_resume_with_amend_after_acknowledge_never_reviews_or_hands_off_that_campaign(
    connect_without_flaws, make_message, script
):
    no_deadline = {**CAMPAIGN_2, "deadline": None}
    client = CampaignRoutingClient(
        {
            (COORDINATOR, None): [
                *coordinator_turns(
                    make_message, [call("c1", "intake")], [call("c2", "review", KEY)]
                ),
                *coordinator_turns(
                    make_message,
                    [call("c1", "intake")],
                    [call("c2", "review", KEY), call("c3", "handoff", KEY)],
                ),
            ],
            (INTAKE, None): 2 * script.intake(CAMPAIGN, no_deadline),
            (REVIEW, "Spring Linen Edit"): script.review(BLOCKING),
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        escalated = await coordinate(client, recorded, TWO_CAMPAIGNS, today=TODAY)
        acknowledged = await resume(client, recorded, escalated.state, Acknowledge(), TODAY)
        state = CoordinatorState.model_validate_json(acknowledged.state.model_dump_json())
        result = await resume(client, recorded, state, Amend(text="Budget £5k."), TODAY)

    calls = client.messages.calls[(COORDINATOR, None)]
    linen, miles = json.loads(calls[4]["messages"][-1]["content"][0]["content"])["campaigns"]
    assert (linen["revisions_requested"], linen["escalated"]) == (True, False)
    assert (miles["revisions_requested"], miles["escalated"]) == (False, True)
    review_denied, handoff_denied = calls[5]["messages"][-1]["content"]
    for denied in (review_denied, handoff_denied):
        assert denied["is_error"] is True
        assert json.loads(denied["content"])["type"] == "permission_denied"
    assert len(client.messages.calls[(REVIEW, "Spring Linen Edit")]) == 2  # first Run only
    assert "tracker_create_task" not in recorded.names
    closed, waiting = result.campaigns
    assert (closed.revisions_requested, closed.handoff) == (True, None)
    assert result.state.escalations == [waiting.escalation]


async def test_resume_with_amend_duplicating_a_closed_key_keeps_that_campaign_closed(
    connect_without_flaws, make_message, script
):
    no_deadline = {**CAMPAIGN_2, "deadline": None}
    client = CampaignRoutingClient(
        {
            (COORDINATOR, None): [
                *coordinator_turns(
                    make_message, [call("c1", "intake")], [call("c2", "review", KEY)]
                ),
                *coordinator_turns(
                    make_message,
                    [call("c1", "intake")],
                    [call("c2", "review", KEY), call("c3", "handoff", KEY)],
                ),
            ],
            (INTAKE, None): [
                *script.intake(CAMPAIGN, no_deadline),
                *script.intake(CAMPAIGN, CAMPAIGN, no_deadline),
            ],
            (REVIEW, "Spring Linen Edit"): script.review(BLOCKING),
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        escalated = await coordinate(client, recorded, TWO_CAMPAIGNS, today=TODAY)
        acknowledged = await resume(client, recorded, escalated.state, Acknowledge(), TODAY)
        result = await resume(
            client, recorded, acknowledged.state, Amend(text="Budget £5k."), TODAY
        )

    calls = client.messages.calls[(COORDINATOR, None)]
    *linens, _ = json.loads(calls[4]["messages"][-1]["content"][0]["content"])["campaigns"]
    assert [c["revisions_requested"] for c in linens] == [True, True]
    review_denied, handoff_denied = calls[5]["messages"][-1]["content"]
    for denied, message in (
        (review_denied, "This campaign is closed: revisions requested."),
        (handoff_denied, "This campaign is escalated to a human, or closed."),
    ):
        assert denied["is_error"] is True
        assert json.loads(denied["content"]) == json.loads(permission_denied(message))
    assert len(client.messages.calls[(REVIEW, "Spring Linen Edit")]) == 2  # first Run only
    assert "tracker_create_task" not in recorded.names
    closed, *duplicates, _ = result.campaigns
    assert closed.revisions_requested is True
    assert len(duplicates) == 2
    for outcome in duplicates:
        assert outcome.escalation.reason == "duplicate_campaign"
        assert "already closed" in outcome.escalation.needs[0]
    assert [e.reason for e in result.state.escalations] == [
        "duplicate_campaign",
        "duplicate_campaign",
        "missing_critical_field",
    ]


async def test_coordinate_review_failing_once_is_retried_and_reaches_handoff(
    connect_without_flaws, routing_client, make_message, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN),
            REVIEW: [make_message("end_turn", text="No review."), *script.review()],
            HANDOFF: script.handoff(),
        }
    )

    async with connect_without_flaws() as session:
        result = await coordinate(client, session, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.escalation is None
    assert outcome.handoff is not None
    assert result.state is None
    assert [(f.agent, f.error) for f in result.failed_runs] == [("review", "NoValidReview")]
    assert result.failed_runs[0].messages[-1]["content"][0].text == "No review."


async def test_coordinate_review_failing_twice_escalates_campaign_accepting_retry(
    connect_without_flaws, routing_client, make_message, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN, ambiguities=["Tone unclear"]),
            REVIEW: [make_message("end_turn", text="No review.")] * 2,
            HANDOFF: [],
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.escalation.reason == "run_failed"
    assert outcome.escalation.answer == "retry"
    assert outcome.escalation.needs == ["Review failed twice (NoValidReview)"]
    assert outcome.escalation.ambiguities == ["Tone unclear"]
    assert len(result.failed_runs) == 2
    refused = client.messages.calls[COORDINATOR][2]["messages"][-1]["content"][0]
    assert refused["is_error"] is True
    assert json.loads(refused["content"])["type"] == "permission_denied"
    assert "tracker_create_task" not in recorded.names
    assert result.state.escalations == [outcome.escalation]


async def test_coordinate_handoff_failing_before_creating_a_task_is_retried(
    connect_without_flaws, routing_client, make_message, script
):
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN),
            REVIEW: script.review(),
            HANDOFF: [make_message("end_turn", text="No handoff."), *script.handoff()],
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    assert outcome.escalation is None
    assert outcome.handoff is not None
    assert recorded.names.count("tracker_create_task") == 1
    assert [f.error for f in result.failed_runs] == ["HandoffIncomplete"]


async def test_coordinate_handoff_failing_after_creating_a_task_escalates_without_retry(
    connect_without_flaws, routing_client, make_message, script
):
    created_only = [
        make_message(
            "tool_use",
            tool_uses=[call("h1", "tracker_create_task", {"title": "T", "description": "D"})],
        ),
        make_message("end_turn", text="Forgot to submit."),
    ]
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN),
            REVIEW: script.review(),
            HANDOFF: created_only,
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(client, recorded, BRIEF, today=TODAY)

    [outcome] = result.campaigns
    [failed] = result.failed_runs
    created = json.loads(failed.messages[2]["content"][0]["content"])  # the create's result
    assert outcome.escalation.reason == "run_failed"
    assert outcome.escalation.answer == "retry"
    assert outcome.escalation.needs == [
        f"Handoff failed (HandoffIncomplete) after creating tracker Task {created['id']}. "
        "Check that Task before retrying."
    ]
    assert recorded.names.count("tracker_create_task") == 1


async def test_resume_with_retry_runs_the_campaign_again_with_a_fresh_retry_count(
    connect_without_flaws, routing_client, make_message, script
):
    full = [call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)]
    no_review = make_message("end_turn", text="No review.")
    client = routing_client(
        {
            COORDINATOR: [*script.coordinator(*full[:2]), *script.coordinator(*full)],
            INTAKE: [*script.intake(CAMPAIGN), *script.intake(CAMPAIGN)],
            # Fails twice, then once more after the retry: a carried count would escalate.
            REVIEW: [no_review, no_review, no_review, *script.review()],
            HANDOFF: script.handoff(),
        }
    )

    async with connect_without_flaws() as session:
        escalated = await coordinate(client, session, BRIEF, today=TODAY)
        state = CoordinatorState.model_validate_json(escalated.state.model_dump_json())
        result = await resume(client, session, state, Retry(), TODAY)

    assert client.messages.calls[INTAKE][2]["messages"][0]["content"].count(BRIEF) == 1
    [outcome] = result.campaigns
    assert outcome.escalation is None
    assert outcome.handoff is not None
    assert [f.error for f in result.failed_runs] == ["NoValidReview"]
    assert result.state is None


class RejectFirstCreate:
    """A session whose first `tracker_create_task` comes back as a server ToolError."""

    def __init__(self, session: Client, error_type: str) -> None:
        self._session = session
        self._error_type = error_type
        self._rejected = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        if name == "tracker_create_task" and not self._rejected:
            self._rejected = True
            error = {
                "type": self._error_type,
                "message": "The tracker could not create the Task.",
                "retryable": self._error_type == "upstream_unavailable",
                "details": {},
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(error))], is_error=True
            )
        return await self._session.call_tool(name, arguments)


@pytest.mark.parametrize(
    ("error_type", "retried"),
    [("invalid_input", True), ("upstream_unavailable", False)],
)
async def test_coordinate_handoff_whose_create_was_rejected_is_retried_only_if_nothing_was_written(
    connect_without_flaws, routing_client, make_message, script, error_type, retried
):
    rejected_create = [
        make_message(
            "tool_use",
            tool_uses=[call("h1", "tracker_create_task", {"title": "T", "description": "D"})],
        ),
        make_message("end_turn", text="Gave up."),
    ]
    client = routing_client(
        {
            COORDINATOR: script.coordinator(
                call("c1", "intake"), call("c2", "review", KEY), call("c3", "handoff", KEY)
            ),
            INTAKE: script.intake(CAMPAIGN),
            REVIEW: script.review(),
            HANDOFF: [*rejected_create, *script.handoff()],
        }
    )

    async with connect_without_flaws() as session:
        recorded = RecordCalls(session)
        result = await coordinate(
            client, RejectFirstCreate(recorded, error_type), BRIEF, today=TODAY
        )

    [outcome] = result.campaigns
    assert [f.error for f in result.failed_runs] == ["HandoffIncomplete"]
    if retried:
        assert outcome.escalation is None
        assert outcome.handoff is not None
        assert recorded.names.count("tracker_create_task") == 1  # the retry's create only
    else:
        assert (outcome.escalation.reason, outcome.escalation.answer) == ("run_failed", "retry")
        assert recorded.names.count("tracker_create_task") == 0  # rejected before the server
        assert outcome.escalation.needs == [
            "Handoff failed (HandoffIncomplete) after trying to create a tracker Task. "
            "Check the tracker for that Task before retrying."
        ]
        assert len(client.messages.calls[HANDOFF]) == 2  # one Handoff Run, not retried
