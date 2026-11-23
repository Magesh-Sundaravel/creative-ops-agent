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
