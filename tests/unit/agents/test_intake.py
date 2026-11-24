from datetime import date
from decimal import Decimal

import pytest

from creative_ops.agents.intake import NoValidSubmission, intake
from creative_ops.schemas.brief import Budget

BRIEF = "Brand: Acme. Launch campaign, two IG reels, deadline 15 Nov 2026, budget £40k."


def submit(tool_id: str, brief: dict | list[dict], ambiguities: list[str] | None = None) -> dict:
    """A `submit_brief` call; a single dict is one campaign."""
    briefs = brief if isinstance(brief, list) else [brief]
    return {
        "id": tool_id,
        "name": "submit_brief",
        "input": {"briefs": briefs, "ambiguities": ambiguities or []},
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

    assert result.campaigns[0].brief.brand == "Acme"
    assert result.campaigns[0].brief.deadline == date(2026, 11, 15)
    assert result.campaigns[0].brief.budget == Budget(amount=Decimal("40000"), currency="GBP")
    assert result.campaigns[0].brief.deliverables[0].channel == "Instagram"
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

    assert result.campaigns[0].brief.brand == "Acme"
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

    assert result.campaigns[0].brief.brand == "Acme Ltd"
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

    assert result.campaigns[0].brief.brand == "Acme Ltd"


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

    assert result.campaigns[0].missing_fields == [
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

    assert "budget.currency" in result.campaigns[0].missing_fields
    assert "budget.amount" not in result.campaigns[0].missing_fields
    assert "deliverables[0].quantity" not in result.campaigns[0].missing_fields
    assert "deliverables[0].format_or_dimensions" in result.campaigns[0].missing_fields
    assert "deliverables[1].channel" in result.campaigns[0].missing_fields
    assert "deliverables[1].quantity" in result.campaigns[0].missing_fields


async def test_empty_deliverables_is_reported_as_deliverables(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, {**COMPLETE, "deliverables": []})

    assert "deliverables" in result.campaigns[0].missing_fields


async def test_complete_critical_fields_do_not_set_critical_missing(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, COMPLETE)

    assert result.campaigns[0].missing_fields  # non-critical gaps are still listed
    assert result.campaigns[0].critical_missing is False


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

    assert result.campaigns[0].critical_missing is True


async def test_critical_fields_lists_only_the_missing_critical_field_paths(
    fake_client, make_message
):
    brief = {**COMPLETE, "deadline": None, "deliverables": [{"asset_type": "reel"}]}

    result = await submit_and_intake(fake_client, make_message, brief)

    assert result.campaigns[0].critical_fields == ["deliverables[0].channel", "deadline"]


async def test_missing_deliverable_due_date_is_not_critical(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, COMPLETE)

    assert "deliverables[0].due_date" in result.campaigns[0].missing_fields
    assert result.campaigns[0].critical_missing is False


async def test_blank_strings_count_as_missing(fake_client, make_message):
    brief = {
        **COMPLETE,
        "objective": "   ",
        "campaign_name": "",
        "deliverables": [{"asset_type": "reel", "channel": ""}],
        "budget": {"amount": "40000", "currency": " "},
    }

    result = await submit_and_intake(fake_client, make_message, brief)

    assert result.campaigns[0].brief.objective is None
    assert {
        "objective",
        "campaign_name",
        "deliverables[0].channel",
        "budget.currency",
    } <= set(result.campaigns[0].missing_fields)
    assert result.campaigns[0].critical_missing is True


async def test_empty_structured_brief_reports_every_field_missing(fake_client, make_message):
    result = await submit_and_intake(fake_client, make_message, {})

    assert result.campaigns[0].missing_fields == [
        "brand",
        "campaign_name",
        "season",
        "objective",
        "target_audience",
        "key_message",
        "usage_rights",
        "deliverables",
        "mandatories",
        "references",
        "deadline",
        "budget",
    ]
    assert result.campaigns[0].critical_missing is True


async def test_text_that_is_not_a_brief_gives_zero_structured_briefs(fake_client, make_message):
    client = fake_client(
        make_message("tool_use", tool_uses=[submit("t1", [], ["Text is not a Brief"])]),
        make_message("end_turn"),
    )

    result = await intake(client, "Hi, what's the weather like?")

    assert result.campaigns == []
    assert result.ambiguities == ["Text is not a Brief"]


async def test_each_campaign_gets_its_own_structured_brief_and_missing_fields(
    fake_client, make_message
):
    spring = {**COMPLETE, "campaign_name": "Spring"}
    summer = {**COMPLETE, "campaign_name": "Summer", "objective": None}
    client = fake_client(
        make_message(
            "tool_use", tool_uses=[submit("t1", [spring, summer], ["Budget split unclear"])]
        ),
        make_message("end_turn"),
    )

    result = await intake(client, "Acme: Spring and Summer campaigns, all due 15 Nov 2026.")

    [first, second] = result.campaigns
    assert (first.brief.campaign_name, second.brief.campaign_name) == ("Spring", "Summer")
    assert first.brief.deadline == second.brief.deadline == date(2026, 11, 15)
    assert "objective" not in first.missing_fields
    assert first.critical_missing is False
    assert "objective" in second.missing_fields
    assert second.critical_missing is True
    assert result.ambiguities == ["Budget split unclear"]


@pytest.mark.parametrize(
    "rule",
    [
        "resolve a date with no year against today's date",
        "next occurrence on or after today",
        "relative or event-based date",
        "budget range",
        "quoting the Brief",
        "past date",
        "never derive the deadline from deliverable due dates",
        "due date is after the deadline",
        "not a Brief",
        "one structured brief per campaign",
        "stated once for all campaigns",
        "campaign name without the season",
    ],
)
async def test_system_prompt_states_the_extraction_rules(fake_client, make_message, rule):
    client = fake_client(make_message("end_turn"))

    with pytest.raises(NoValidSubmission):
        await intake(client, BRIEF)

    assert rule.lower() in client.messages.calls[0]["system"].lower()
