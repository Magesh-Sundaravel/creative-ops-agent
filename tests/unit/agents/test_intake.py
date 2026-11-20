from datetime import date
from decimal import Decimal

import pytest

from creative_ops.agents.intake import NoValidSubmission, intake
from creative_ops.schemas.brief import Budget

BRIEF = "Brand: Acme. Launch campaign, two IG reels, deadline 15 Nov 2026, budget £40k."


def submit(tool_id: str, brief: dict, ambiguities: list[str] | None = None) -> dict:
    return {
        "id": tool_id,
        "name": "submit_brief",
        "input": {"brief": brief, "ambiguities": ambiguities or []},
    }


async def test_valid_submission_returns_structured_brief_and_ambiguities(fake_client, make_message):
    client = fake_client(
        make_message(
            "tool_use",
            tool_uses=[
                submit(
                    "t1",
                    {
                        "brand": "Acme",
                        "deadline": "2026-11-15",
                        "budget": {"amount": "40000", "currency": "GBP"},
                        "deliverables": [{"asset_type": "reel", "channel": "Instagram"}],
                    },
                    ["Quantity of reels unclear"],
                )
            ],
        ),
        make_message("end_turn", text="Done"),
    )

    result = await intake(client, BRIEF, today=date(2026, 11, 1))

    assert result.brief.brand == "Acme"
    assert result.brief.deadline == date(2026, 11, 15)
    assert result.brief.budget == Budget(amount=Decimal("40000"), currency="GBP")
    assert result.brief.deliverables[0].channel == "Instagram"
    assert result.ambiguities == ["Quantity of reels unclear"]
    assert result.loop_result.text == "Done"


async def test_invalid_submission_is_returned_as_error_and_corrected_one_succeeds(
    fake_client, make_message
):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", {"deadline": "soon"})]),
        make_message("tool_use", tool_uses=[submit("t2", {"brand": "Acme"})]),
        make_message("end_turn"),
    )

    result = await intake(client, BRIEF)

    assert result.brief.brand == "Acme"
    [bad_result] = client.messages.calls[1]["messages"][-1]["content"]
    assert bad_result["tool_use_id"] == "t1"
    assert bad_result["is_error"] is True


@pytest.mark.parametrize("first_reply", ["end_turn", "invalid"])
async def test_run_without_valid_submission_raises_with_trace(
    fake_client, make_message, first_reply
):
    if first_reply == "end_turn":
        client = fake_client(make_message("end_turn", text="I can't"))
    else:
        client = fake_client(
            make_message("tool_use", tool_uses=[submit("t1", {"deadline": "soon"})]),
            make_message("end_turn", text="Giving up"),
        )

    with pytest.raises(NoValidSubmission) as exc_info:
        await intake(client, BRIEF)

    assert exc_info.value.loop_result.messages[0] == {"role": "user", "content": BRIEF}


async def test_last_valid_submission_wins_across_iterations(fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", {"brand": "Acme"}, ["first"])]),
        make_message("tool_use", tool_uses=[submit("t2", {"brand": "Acme Ltd"}, ["second"])]),
        make_message("end_turn"),
    )

    result = await intake(client, BRIEF)

    assert result.brief.brand == "Acme Ltd"
    assert result.ambiguities == ["second"]


async def test_last_valid_submission_in_call_order_wins_within_one_response(
    fake_client, make_message
):
    client = fake_client(
        make_message(
            "tool_use",
            tool_uses=[
                submit("t1", {"brand": "Acme"}),
                submit("t2", {"brand": "Acme Ltd"}),
                submit("t3", {"deadline": "soon"}),
            ],
        ),
        make_message("end_turn"),
    )

    result = await intake(client, BRIEF)

    assert result.brief.brand == "Acme Ltd"


async def test_missing_fields_lists_every_field_the_brief_does_not_give(fake_client, make_message):
    client = fake_client(
        make_message(
            "tool_use",
            tool_uses=[
                submit(
                    "t1",
                    {
                        "brand": "Acme",
                        "deadline": "2026-11-15",
                        "deliverables": [{"asset_type": "reel"}],
                        "mandatories": ["logo"],
                    },
                )
            ],
        ),
        make_message("end_turn"),
    )

    result = await intake(client, BRIEF)

    assert result.missing_fields == [
        "campaign_name",
        "season",
        "objective",
        "target_audience",
        "key_message",
        "usage_rights",
        "deliverables[0].channel",
        "deliverables[0].format_or_dimensions",
        "deliverables[0].quantity",
        "deliverables[0].due_date",
        "references",
        "budget",
    ]


async def test_today_is_put_in_the_system_prompt(fake_client, make_message):
    client = fake_client(make_message("end_turn"))

    with pytest.raises(NoValidSubmission):
        await intake(client, BRIEF, today=date(2026, 11, 1))

    assert "2026-11-01" in client.messages.calls[0]["system"]


async def test_today_defaults_to_the_real_date(fake_client, make_message):
    client = fake_client(make_message("end_turn"))

    with pytest.raises(NoValidSubmission):
        await intake(client, BRIEF)

    assert date.today().isoformat() in client.messages.calls[0]["system"]


COMPLETE = {
    "brand": "Acme",
    "objective": "Awareness",
    "deadline": "2026-11-15",
    "deliverables": [{"asset_type": "reel", "channel": "Instagram"}],
}


async def submit_and_intake(fake_client, make_message, brief: dict):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", brief)]), make_message("end_turn")
    )
    return await intake(client, BRIEF)


async def test_nested_gaps_are_reported_as_paths(fake_client, make_message):
    brief = {
        **COMPLETE,
        "deliverables": [
            {"asset_type": "reel", "channel": "Instagram", "quantity": 2},
            {"asset_type": "still"},
        ],
        "budget": {"amount": "40000"},
    }

    result = await submit_and_intake(fake_client, make_message, brief)

    assert "budget.currency" in result.missing_fields
    assert "budget.amount" not in result.missing_fields
    assert "deliverables[0].quantity" not in result.missing_fields
    assert "deliverables[0].format_or_dimensions" in result.missing_fields
    assert "deliverables[1].channel" in result.missing_fields
    assert "deliverables[1].quantity" in result.missing_fields


async def test_empty_deliverables_is_reported_as_deliverables(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, {**COMPLETE, "deliverables": []})

    assert "deliverables" in result.missing_fields


async def test_complete_critical_fields_do_not_set_critical_missing(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, COMPLETE)

    assert result.missing_fields  # non-critical gaps are still listed
    assert result.critical_missing is False


@pytest.mark.parametrize(
    "brief",
    [
        {**COMPLETE, "brand": None},
        {**COMPLETE, "objective": None},
        {**COMPLETE, "deadline": None},
        {**COMPLETE, "deliverables": []},
        {**COMPLETE, "deliverables": [{"channel": "Instagram"}]},
        {**COMPLETE, "deliverables": [{"asset_type": "reel"}]},
    ],
    ids=["brand", "objective", "deadline", "no-deliverables", "asset_type", "channel"],
)
async def test_missing_critical_field_sets_critical_missing(fake_client, make_message, brief):
    result = await submit_and_intake(fake_client, make_message, brief)

    assert result.critical_missing is True


async def test_missing_deliverable_due_date_is_not_critical(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, COMPLETE)

    assert "deliverables[0].due_date" in result.missing_fields
    assert result.critical_missing is False
