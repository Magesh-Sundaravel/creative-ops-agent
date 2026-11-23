import pytest

from creative_ops.agents.review import CannotReview, NoValidReview, review
from creative_ops.schemas.brief import Deliverable, StructuredBrief

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


async def test_review_advisory_only_findings_returns_approved(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [finding("AST-001", "advisory")])]),
        make_message("end_turn", text="Done"),
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    assert result.verdict == "approved"
    assert result.brand == "loomhouse"
    assert [f.subject for f in result.findings] == ["AST-001"]
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
    assert [f.subject for f in result.findings] == ["AST-001", 1]


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


async def test_review_last_valid_submission_wins(connect, fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [finding(1, "blocking")])]),
        make_message("tool_use", tool_uses=[submit("t2", [finding("AST-006", "advisory")])]),
        make_message("end_turn"),
    )

    async with connect() as session:
        result = await review(client, session, BRIEF)

    assert [f.subject for f in result.findings] == ["AST-006"]
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

    assert [f.subject for f in result.findings] == [1]
    assert result.verdict == "revisions_requested"
