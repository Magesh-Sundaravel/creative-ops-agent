import json
from datetime import date

import pytest
from mcp.types import CallToolResult, TextContent

from creative_ops.agents.handoff import CannotHandoff, HandoffIncomplete, handoff
from creative_ops.agents.review import Finding, ReviewResult
from creative_ops.loop.agent_loop import LoopResult, StopOutcome, TokenUsage
from creative_ops.schemas.brief import Deliverable, StructuredBrief

BRIEF = StructuredBrief(
    brand="Loomhouse",
    campaign_name="Spring Linen Edit",
    deliverables=[
        Deliverable(asset_type="image", channel="Instagram"),
        Deliverable(asset_type="video", channel="Facebook"),
    ],
    deadline=date(2031, 3, 1),
)
ADVISORY = Finding(
    subject="AST-004",
    rule="tone",
    explanation="Copy is louder than the brand's tone of voice.",
    severity="advisory",
    source="judgement",
)
REVIEW = ReviewResult(
    brand="loomhouse",
    findings=[ADVISORY],
    coverage={0: ["AST-001", "AST-004"], 1: ["AST-008"]},
    verdict="approved",
    loop_result=LoopResult(
        text="",
        outcome=StopOutcome.COMPLETED,
        messages=[],
        usage=TokenUsage(input_tokens=0, output_tokens=0),
    ),
)


@pytest.mark.parametrize(
    "update",
    [{"verdict": "revisions_requested"}, {"coverage": {}}],
    ids=["not-approved", "no-covering-assets"],
)
async def test_handoff_review_not_approved_or_covering_nothing_raises_without_model_call(
    connect, fake_client, update
):
    client = fake_client()
    review = REVIEW.model_copy(update=update)

    async with connect() as session:
        with pytest.raises(CannotHandoff):
            await handoff(client, session, BRIEF, review)

    assert client.messages.calls == []


def create(tool_id: str, title: str = "Spring Linen Edit: social set") -> dict:
    return {
        "id": tool_id,
        "name": "tracker_create_task",
        "input": {"title": title, "description": "Deliver the feed set and the 20s video."},
    }


def submit(tool_id: str, notes: str = "Lead with the outdoor shot.") -> dict:
    return {"id": tool_id, "name": "submit_handoff", "input": {"production_notes": notes}}


def tool_results(client, call: int) -> list[dict]:
    return client.messages.calls[call]["messages"][-1]["content"]


async def test_handoff_creates_task_with_code_filled_fields_and_returns_its_id(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1")]),
        make_message("tool_use", tool_uses=[submit("t2")]),
        make_message("end_turn", text="Done"),
    )

    async with connect() as session:
        package = await handoff(client, session, BRIEF, REVIEW)
        # No read tool exists; setting the unchanged status returns the Task as stored.
        stored = await session.call_tool(
            "tracker_update_status", {"task_id": package.task_id, "status": "briefed"}
        )

    assert stored.structured_content == {
        "id": package.task_id,
        "title": "Spring Linen Edit: social set",
        "brand": "loomhouse",
        "campaign_name": "Spring Linen Edit",
        "description": "Deliver the feed set and the 20s video.",
        "due_date": "2031-03-01",
        "status": "briefed",
    }
    assert package.production_notes == "Lead with the outdoor shot."
    assert package.loop_result.text == "Done"


async def test_handoff_package_carries_facts_assembled_from_the_client_stack(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1"), submit("t2")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        package = await handoff(client, session, BRIEF, REVIEW)

    assert package.brand == "loomhouse"
    assert package.campaign_name == "Spring Linen Edit"
    assert [
        (d.deliverable.channel, [(a.id, a.file_name) for a in d.assets])
        for d in package.deliverables
    ] == [
        (
            "Instagram",
            [
                ("AST-001", "loomhouse_spring-linen_ig-feed_01.jpg"),
                ("AST-004", "loomhouse_spring-linen_ig-feed_02.jpg"),
            ],
        ),
        ("Facebook", [("AST-008", "loomhouse_spring-linen_fb-feed_20s.mp4")]),
    ]
    assert package.legal_lines == ["Loomhouse is a trading name of Loomhouse Ltd."]
    assert package.usage_rights.model_dump() == {
        "territories": ["UK", "EU"],
        "channels": ["instagram", "facebook"],
        "expires_on": date(2031, 3, 31),
    }
    assert package.deadline == date(2031, 3, 1)
    assert package.notes == [ADVISORY]


async def test_handoff_usage_rights_summary_keeps_only_what_every_covering_asset_shares(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1"), submit("t2")]),
        make_message("end_turn"),
    )
    # AST-001 covers instagram and facebook; AST-006 covers website and email.
    review = REVIEW.model_copy(update={"coverage": {0: ["AST-001"], 1: ["AST-006"]}})

    async with connect() as session:
        package = await handoff(client, session, BRIEF, review)

    assert package.usage_rights.territories == ["UK", "EU"]
    assert package.usage_rights.channels == []


@pytest.mark.parametrize("calls", [[create("t1")], [submit("t1")]], ids=["no-submit", "no-create"])
async def test_handoff_run_without_create_and_submission_raises_with_trace(
    connect, fake_client, make_message, calls
):
    client = fake_client(
        make_message("tool_use", tool_uses=calls), make_message("end_turn", text="Giving up")
    )

    async with connect() as session:
        with pytest.raises(HandoffIncomplete) as exc_info:
            await handoff(client, session, BRIEF, REVIEW)

    assert exc_info.value.loop_result.text == "Giving up"
    assert exc_info.value.loop_result.messages[1]["content"][0].name == calls[0]["name"]


def get_asset(tool_id: str, asset_id: str) -> dict:
    return {"id": tool_id, "name": "dam_get_asset", "input": {"asset_id": asset_id}}


async def test_handoff_offers_model_only_its_allowlisted_tools(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1"), submit("t2")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        await handoff(client, session, BRIEF, REVIEW)
        server_tools = (await session.list_tools()).tools

    offered = {t["name"]: t for t in client.messages.calls[0]["tools"]}
    assert sorted(offered) == ["dam_get_asset", "submit_handoff", "tracker_create_task"]
    [server_get_asset] = [t for t in server_tools if t.name == "dam_get_asset"]
    assert offered["dam_get_asset"]["input_schema"] == server_get_asset.input_schema
    assert set(offered["tracker_create_task"]["input_schema"]["properties"]) == {
        "title",
        "description",
    }


async def test_handoff_model_can_fetch_an_asset_through_mcp(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[get_asset("t1", "AST-008")]),
        make_message("tool_use", tool_uses=[create("t2"), submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        await handoff(client, session, BRIEF, REVIEW)

    [result] = tool_results(client, 1)
    assert "is_error" not in result
    assert json.loads(result["content"])["file_name"] == "loomhouse_spring-linen_fb-feed_20s.mp4"


async def test_handoff_server_error_reaches_model_as_is_error_tool_error(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[get_asset("t1", "AST-999")]),
        make_message("tool_use", tool_uses=[create("t2"), submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        await handoff(client, session, BRIEF, REVIEW)

    [result] = tool_results(client, 1)
    assert result["is_error"] is True
    error = json.loads(result["content"])
    assert error["type"] == "not_found"
    assert error["details"] == {"asset_id": "AST-999"}


class RecordCalls:
    """A session that records the name of every tool call it passes to the server."""

    def __init__(self, session) -> None:
        self._session = session
        self.called: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict):
        self.called.append(name)
        return await self._session.call_tool(name, arguments)


async def test_handoff_gate_allows_one_create_when_two_arrive_in_one_response(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1"), create("t2", title="Duplicate")]),
        make_message("tool_use", tool_uses=[submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        recorded = RecordCalls(session)
        await handoff(client, recorded, BRIEF, REVIEW)

    first, second = tool_results(client, 1)
    assert "is_error" not in first
    assert second["is_error"] is True
    assert json.loads(second["content"])["type"] == "permission_denied"
    assert recorded.called.count("tracker_create_task") == 1


async def test_handoff_gate_denies_a_create_in_a_later_iteration(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1")]),
        make_message("tool_use", tool_uses=[create("t2", title="Again"), submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        recorded = RecordCalls(session)
        await handoff(client, recorded, BRIEF, REVIEW)

    second, _ = tool_results(client, 2)
    assert second["is_error"] is True
    assert json.loads(second["content"])["type"] == "permission_denied"
    assert recorded.called.count("tracker_create_task") == 1


class RejectFirstCreate:
    """A session whose first `tracker_create_task` comes back as a server ToolError."""

    def __init__(self, session, error_type: str) -> None:
        self._session = session
        self._error_type = error_type
        self._rejected = False

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict):
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


async def test_handoff_gate_releases_its_slot_when_the_server_rejects_a_create(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1")]),
        make_message("tool_use", tool_uses=[create("t2"), submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        recorded = RecordCalls(session)
        await handoff(client, RejectFirstCreate(recorded, "invalid_input"), BRIEF, REVIEW)

    [rejected] = tool_results(client, 1)
    assert rejected["is_error"] is True
    assert json.loads(rejected["content"])["type"] == "invalid_input"
    retried, _ = tool_results(client, 2)
    assert "is_error" not in retried
    assert recorded.called.count("tracker_create_task") == 1


async def test_handoff_gate_keeps_its_slot_when_the_create_outcome_is_unknown(
    connect, fake_client, make_message
):
    # upstream_unavailable may mean the Task was written and only the reply was lost.
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1")]),
        make_message("tool_use", tool_uses=[create("t2"), submit("t3")]),
        make_message("end_turn"),
    )

    async with connect() as session:
        recorded = RecordCalls(session)
        with pytest.raises(HandoffIncomplete):
            await handoff(
                client, RejectFirstCreate(recorded, "upstream_unavailable"), BRIEF, REVIEW
            )

    [unavailable] = tool_results(client, 1)
    assert json.loads(unavailable["content"])["type"] == "upstream_unavailable"
    retried, _ = tool_results(client, 2)
    assert json.loads(retried["content"])["type"] == "permission_denied"
    assert recorded.called.count("tracker_create_task") == 0


async def test_handoff_gate_denies_a_create_when_the_verdict_is_not_approved(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1")]),
        make_message("end_turn"),
    )
    review = REVIEW.model_copy()
    scripted = client.messages.create

    async def bypass_precondition(**kwargs):
        # The agent has already checked the verdict; change it under the Gate's feet.
        review.verdict = "revisions_requested"
        return await scripted(**kwargs)

    client.messages.create = bypass_precondition

    async with connect() as session:
        recorded = RecordCalls(session)
        with pytest.raises(HandoffIncomplete):
            await handoff(client, recorded, BRIEF, review)

    [denied] = tool_results(client, 1)
    assert denied["is_error"] is True
    assert json.loads(denied["content"])["type"] == "permission_denied"
    assert "tracker_create_task" not in recorded.called


async def test_handoff_model_cannot_change_a_task_status(connect, fake_client, make_message):
    update = {"task_id": "TSK-001", "status": "delivered"}
    client = fake_client(
        make_message("tool_use", tool_uses=[create("t1"), submit("t2")]),
        make_message(
            "tool_use",
            tool_uses=[{"id": "t3", "name": "tracker_update_status", "input": update}],
        ),
        make_message("end_turn"),
    )

    async with connect() as session:
        recorded = RecordCalls(session)
        await handoff(client, recorded, BRIEF, REVIEW)

    [result] = tool_results(client, 2)
    assert result["is_error"] is True
    assert result["content"].startswith("Unknown tool 'tracker_update_status'")
    assert "tracker_update_status" not in recorded.called
