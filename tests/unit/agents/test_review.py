import json
import shutil
from datetime import date

import pytest
from mcp import Client

from creative_ops.agents.review import CannotReview, NoValidReview, review
from creative_ops.schemas.brief import Deliverable, StructuredBrief
from mcp_server.server import SEED_DIR, build_server

BRIEF = StructuredBrief(
    brand="loomhouse",
    campaign_name="Spring Linen Edit",
    objective="Launch the linen range",
    deliverables=[
        Deliverable(asset_type="image", channel="Instagram"),
        Deliverable(asset_type="image", channel="Website"),
    ],
)
COVERAGE = {"0": ["AST-001"], "1": ["AST-006"]}


@pytest.fixture
def connect_without_flaws(tmp_path):
    """Like `connect`, but the seed has no Spring Linen Edit Asset that fails a check."""
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    assets = json.loads((tmp_path / "assets.json").read_text())
    (tmp_path / "assets.json").write_text(json.dumps([a for a in assets if a["id"] != "AST-009"]))
    return lambda: Client(build_server(tmp_path))


def judged(result) -> list:
    return [f.subject for f in result.findings if f.source == "judgement"]


def submit(tool_id: str, findings: list[dict], coverage: dict | None = None) -> dict:
    return {
        "id": tool_id,
        "name": "submit_review",
        "input": {"findings": findings, "coverage": COVERAGE if coverage is None else coverage},
    }


def finding(subject: str | int, severity: str) -> dict:
    return {
        "subject": subject,
        "rule": "tone",
        "explanation": "Copy is louder than the brand's tone of voice.",
        "severity": severity,
    }


async def test_review_advisory_only_findings_returns_approved(
    connect_without_flaws, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [finding("AST-001", "advisory")])]),
        make_message("end_turn", text="Done"),
    )

    async with connect_without_flaws() as session:
        result = await review(client, session, BRIEF)

    assert result.verdict == "approved"
    assert result.brand == "loomhouse"
    assert [(f.subject, f.source) for f in result.findings] == [("AST-001", "judgement")]
    assert result.coverage == {0: ["AST-001"], 1: ["AST-006"]}
    assert result.loop_result.text == "Done"


async def test_review_any_blocking_finding_returns_revisions_requested(
    connect, fake_client, make_message
):
    findings = [finding("AST-001", "advisory"), finding(1, "blocking")]
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", findings, {"0": ["AST-001"], "1": []})]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    assert result.verdict == "revisions_requested"
    assert judged(result) == ["AST-001", 1]


@pytest.mark.parametrize(
    ("brand", "campaign", "slug", "asset_id"),
    [
        ("Loomhouse", "Spring Linen Edit", "loomhouse", "AST-001"),
        ("LOOMHOUSE", "Spring Linen Edit", "loomhouse", "AST-001"),
        ("verdant skin", "Dew Ritual", "verdant-skin", "AST-003"),
    ],
)
async def test_review_brand_in_any_case_resolves_to_slug(
    connect, fake_client, make_message, brand, campaign, slug, asset_id
):
    brief = BRIEF.model_copy(
        update={"brand": brand, "campaign_name": campaign, "deliverables": BRIEF.deliverables[:1]}
    )
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [], {"0": [asset_id]})]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, brief)

    assert result.brand == slug


@pytest.mark.parametrize(
    "update",
    [
        {"brand": None},
        {"campaign_name": None},
        {"brand": "Loomhous"},
        {"campaign_name": "Autumn Wool Edit"},
    ],
    ids=["no-brand", "no-campaign", "unknown-brand", "no-assets"],
)
async def test_review_unreviewable_brief_raises_without_model_call(connect, fake_client, update):
    client = fake_client()

    async with connect() as session:
        with pytest.raises(CannotReview):
            await review(client, session, BRIEF.model_copy(update=update))

    assert client.messages.calls == []


async def test_review_run_without_valid_submission_raises_with_trace(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [], {"0": ["AST-001"]})]),
        make_message("end_turn", text="Giving up"),
    )

    async with connect() as session:
        with pytest.raises(NoValidReview) as exc_info:
            await review(client, session, BRIEF)

    assert exc_info.value.loop_result.text == "Giving up"
    assert exc_info.value.loop_result.messages[1]["content"][0].name == "submit_review"


async def test_review_offers_model_only_submit_review(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [])]), make_message("end_turn")
    )

    async with connect() as session:
        await review(client, session, BRIEF)

    assert [t["name"] for t in client.messages.calls[0]["tools"]] == ["submit_review"]


@pytest.mark.parametrize(
    ("findings", "coverage", "error"),
    [
        ([], {"0": ["AST-001"]}, "deliverable indices"),
        ([], {"0": ["AST-001"], "1": ["AST-002"]}, "not under review"),
        ([], {"0": ["AST-001"], "1": []}, "no covering Asset"),
        ([finding("AST-099", "blocking")], None, "subjects"),
        ([finding(2, "blocking")], None, "subjects"),
    ],
    ids=[
        "missing-deliverable",
        "unknown-asset",
        "uncovered-without-finding",
        "unknown-asset-subject",
        "out-of-range-deliverable-subject",
    ],
)
async def test_review_bad_submission_is_returned_as_error_and_corrected_one_wins(
    connect, fake_client, make_message, findings, coverage, error
):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", findings, coverage)]),
        make_message("tool_use", tool_uses=[submit("t2", [])]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    [bad_result] = client.messages.calls[1]["messages"][-1]["content"]
    assert bad_result["is_error"] is True
    assert error in bad_result["content"]
    assert result.coverage == {0: ["AST-001"], 1: ["AST-006"]}


async def test_review_last_valid_submission_wins(connect_without_flaws, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [finding(1, "blocking")])]),
        make_message("tool_use", tool_uses=[submit("t2", [finding("AST-006", "advisory")])]),
        make_message("end_turn"),
    )

    async with connect_without_flaws() as session:
        result = await review(client, session, BRIEF)

    assert judged(result) == ["AST-006"]
    assert result.verdict == "approved"


async def test_review_deliverable_index_given_as_string_counts_as_that_deliverable(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message(
            "tool_use",
            tool_uses=[submit("t1", [finding("1", "blocking")], {"0": ["AST-001"], "1": []})],
        ),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    assert judged(result) == [1]


async def test_review_draft_asset_is_blocking_check_finding_without_model_findings(
    connect, fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [])]), make_message("end_turn")
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    checks = [(f.subject, f.rule, f.severity) for f in result.findings if f.source == "check"]
    assert checks == [("AST-009", "status", "blocking")]
    assert result.verdict == "revisions_requested"


@pytest.mark.parametrize(
    ("brand", "campaign", "asset_id", "expected"),
    [
        ("loomhouse", "Spring Linen Edit", "AST-001", [("AST-009", "status")]),
        ("stridewell", "City Miles", "AST-002", [("AST-014", "dimensions")]),
        (
            "verdant-skin",
            "Dew Ritual",
            "AST-003",
            [("AST-019", "approved_channel"), ("AST-020", "rights_expiry")],
        ),
    ],
)
async def test_review_seed_flaws_are_the_only_blocking_check_findings(
    connect, fake_client, make_message, brand, campaign, asset_id, expected
):
    brief = BRIEF.model_copy(
        update={"brand": brand, "campaign_name": campaign, "deliverables": BRIEF.deliverables[:1]}
    )
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [], {"0": [asset_id]})]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, brief)

    checks = [f for f in result.findings if f.source == "check"]
    assert [(f.subject, f.rule) for f in checks] == expected
    assert {f.severity for f in checks} == {"blocking"}
    assert result.verdict == "revisions_requested"


@pytest.mark.parametrize(
    ("deadline", "today", "expired"),
    [
        (None, date(2025, 6, 30), False),
        (None, date(2025, 7, 1), True),
        (date(2025, 6, 30), date(2025, 1, 1), False),
        (date(2025, 7, 1), date(2025, 1, 1), True),
        (date(2025, 6, 30), date(2026, 1, 1), False),
    ],
    ids=[
        "today-on-expiry",
        "today-after-expiry",
        "deadline-on-expiry",
        "deadline-after-expiry",
        "deadline-wins-over-later-today",
    ],
)
async def test_review_rights_expiry_against_deadline_or_today(
    connect, fake_client, make_message, deadline, today, expired
):
    # AST-020's Usage Rights expire on 2025-06-30.
    brief = BRIEF.model_copy(
        update={
            "brand": "verdant-skin",
            "campaign_name": "Dew Ritual",
            "deliverables": BRIEF.deliverables[:1],
            "deadline": deadline,
        }
    )
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [], {"0": ["AST-016"]})]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, brief, today=today)

    expiry = [f.subject for f in result.findings if f.rule == "rights_expiry"]
    assert expiry == (["AST-020"] if expired else [])


async def test_review_gives_model_the_check_findings_as_facts(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [])]), make_message("end_turn")
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    [check] = [f for f in result.findings if f.source == "check"]
    assert check.explanation in client.messages.calls[0]["messages"][0]["content"]


async def test_review_model_cannot_downgrade_or_claim_a_check_finding(
    connect, fake_client, make_message
):
    softened = {**finding("AST-009", "advisory"), "rule": "status", "source": "check"}
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [softened])]), make_message("end_turn")
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    assert [(f.subject, f.severity, f.source) for f in result.findings] == [
        ("AST-009", "blocking", "check"),
        ("AST-009", "advisory", "judgement"),
    ]
    assert result.verdict == "revisions_requested"
