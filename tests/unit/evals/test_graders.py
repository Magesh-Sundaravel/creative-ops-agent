from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from creative_ops.agents.coordinator import (
    CampaignOutcome,
    CoordinatorResult,
    Escalation,
    EscalationReason,
)
from creative_ops.agents.handoff import HandoffPackage, RightsSummary
from creative_ops.agents.intake import CampaignIntake, IntakeResult
from creative_ops.loop.agent_loop import LoopResult, StopOutcome, TokenUsage
from creative_ops.schemas.brief import Budget, Deliverable, StructuredBrief
from evals.cases import EvalCase, ExpectedCampaign
from evals.graders import grade_case

LOOP = LoopResult(
    text="",
    outcome=StopOutcome.COMPLETED,
    messages=[],
    usage=TokenUsage(input_tokens=0, output_tokens=0),
)


def eval_case(
    escalation: EscalationReason | None = None,
    campaigns: Sequence[ExpectedCampaign] = (),
    briefs: Sequence[StructuredBrief] = (),
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case",
            "brief": "A Brief.",
            "today": "2031-10-01",
            "tags": [],
            "expected": {
                "briefs": [
                    b.model_dump(mode="json")
                    for b in briefs
                    or [
                        StructuredBrief(brand=c.brand, campaign_name=c.campaign_name)
                        for c in campaigns
                    ]
                ],
                "ambiguities": [],
                "escalation": escalation,
                "campaigns": list(campaigns),
            },
        }
    )


def escalation(reason: EscalationReason) -> Escalation:
    return Escalation(reason=reason, answer="amend", needs=[], ambiguities=[])


def package(brand: str, campaign_name: str) -> HandoffPackage:
    return HandoffPackage(
        brand=brand,
        campaign_name=campaign_name,
        deliverables=[],
        legal_lines=[],
        usage_rights=RightsSummary(territories=[], channels=[], expires_on=date(2032, 9, 30)),
        deadline=None,
        notes=[],
        production_notes="",
        task_id="TSK-100",
        loop_result=LOOP,
    )


def intake(briefs: Sequence[StructuredBrief]) -> IntakeResult:
    campaigns = [
        CampaignIntake(brief=b, missing_fields=[], critical_fields=[], critical_missing=False)
        for b in briefs
    ]
    return IntakeResult(campaigns=campaigns, ambiguities=[], loop_result=LOOP)


def result(
    campaigns: Sequence[CampaignOutcome] = (),
    brief_escalation: Escalation | None = None,
    briefs: Sequence[StructuredBrief] | None = None,
) -> CoordinatorResult:
    return CoordinatorResult(
        campaigns=list(campaigns),
        escalation=brief_escalation,
        intakes=[] if briefs is None else [intake(briefs)],
        reviews=[],
        failed_runs=[],
        loop_result=LOOP,
        state=None,
    )


WINTER = ExpectedCampaign(brand="Loomhouse", campaign_name="Winter Knit Edit", outcome="handed_off")


def outcome_grades(grades):
    return [(g.campaign, g.field, g.passed) for g in grades.grades if g.grader == "outcome"]


async def test_grade_case_matching_outcome_passes(fake_client):
    handed_off = CampaignOutcome(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        handoff=package("loomhouse", "Winter Knit Edit"),
    )

    grades = await grade_case(eval_case(campaigns=[WINTER]), result([handed_off]), fake_client())

    assert outcome_grades(grades) == [
        (None, "brief_escalation", True),
        ("Loomhouse / Winter Knit Edit", "campaign_outcome", True),
    ]
    assert grades.passed


async def test_grade_case_pairs_campaigns_ignoring_case_and_surrounding_whitespace(fake_client):
    handed_off = CampaignOutcome(
        brand=" loomhouse",
        campaign_name="WINTER KNIT EDIT ",
        handoff=package("loomhouse", "WINTER KNIT EDIT"),
    )

    grades = await grade_case(eval_case(campaigns=[WINTER]), result([handed_off]), fake_client())

    assert ("Loomhouse / Winter Knit Edit", "campaign_outcome", True) in outcome_grades(grades)


async def test_grade_case_wrong_brief_level_escalation_reason_fails(fake_client):
    produced = result(brief_escalation=escalation("run_failed"))

    grades = await grade_case(eval_case(escalation="not_a_brief"), produced, fake_client())

    assert outcome_grades(grades) == [(None, "brief_escalation", False)]
    assert not grades.passed


async def test_grade_case_wrong_per_campaign_outcome_fails(fake_client):
    expected = ExpectedCampaign(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        outcome="escalated",
        reason="missing_critical_field",
    )
    revisions = CampaignOutcome(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        escalation=escalation("revisions_requested"),
    )

    grades = await grade_case(eval_case(campaigns=[expected]), result([revisions]), fake_client())

    [(_, _, passed)] = [g for g in outcome_grades(grades) if g[1] == "campaign_outcome"]
    assert not passed
    assert not grades.passed


async def test_grade_case_escalated_with_the_wrong_reason_fails(fake_client):
    expected = ExpectedCampaign(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        outcome="escalated",
        reason="missing_critical_field",
    )
    escalated = CampaignOutcome(
        brand="Loomhouse", campaign_name="Winter Knit Edit", escalation=escalation("cannot_review")
    )

    grades = await grade_case(eval_case(campaigns=[expected]), result([escalated]), fake_client())

    assert not grades.passed


def unfinished(brand: str, campaign_name: str) -> CampaignOutcome:
    return CampaignOutcome(brand=brand, campaign_name=campaign_name)


def count_grade(grades):
    [grade] = [g for g in grades.grades if g.field == "campaign_count"]
    return grade


async def test_grade_case_same_campaigns_pass_the_count_check(fake_client):
    produced = result([unfinished("loomhouse ", "Winter Knit Edit")])

    grades = await grade_case(eval_case(campaigns=[WINTER]), produced, fake_client())

    assert count_grade(grades).passed


async def test_grade_case_missed_campaign_fails_the_count_check(fake_client):
    grades = await grade_case(eval_case(campaigns=[WINTER]), result(), fake_client())

    grade = count_grade(grades)
    assert not grade.passed
    assert "missed" in grade.reason
    assert not grades.passed


async def test_grade_case_spurious_campaign_fails_the_count_check(fake_client):
    produced = result(
        [unfinished("Loomhouse", "Winter Knit Edit"), unfinished("Loomhouse", "Extra")]
    )

    grades = await grade_case(eval_case(campaigns=[WINTER]), produced, fake_client())

    grade = count_grade(grades)
    assert not grade.passed
    assert "spurious" in grade.reason


async def test_grade_case_two_campaigns_with_one_key_fail_the_count_check(fake_client):
    produced = result(
        [unfinished("Loomhouse", "Winter Knit Edit"), unfinished("LOOMHOUSE", "winter knit edit")]
    )

    grades = await grade_case(eval_case(campaigns=[WINTER]), produced, fake_client())

    assert not count_grade(grades).passed


WINTER_BRIEF = StructuredBrief.model_validate(
    {
        "brand": "Loomhouse",
        "campaign_name": "Winter Knit Edit",
        "season": "AW31",
        "objective": "Launch the winter knitwear range",
        "deliverables": [
            {"asset_type": "image", "channel": "instagram", "format_or_dimensions": "1080x1350"},
            {"asset_type": "image", "channel": "instagram", "format_or_dimensions": "1080x1920"},
            {
                "asset_type": "image",
                "channel": "email",
                "format_or_dimensions": "1200x600",
                "quantity": 1,
                "due_date": "2031-11-01",
            },
        ],
        "mandatories": ["Show the wool fibre up close", "Logo bottom right"],
        "references": ["AW30 lookbook", "Mood board v2"],
        "deadline": "2031-11-15",
        "budget": {"amount": "8000", "currency": "GBP"},
    }
)


def field_grades(grades):
    return [(g.campaign, g.field, g.passed) for g in grades.grades if g.grader == "fields"]


def failed_fields(grades):
    return [field for _, field, passed in field_grades(grades) if not passed]


async def grade_brief(fake_client, produced: StructuredBrief):
    produced_campaign = unfinished(produced.brand, produced.campaign_name)
    return await grade_case(
        eval_case(campaigns=[WINTER], briefs=[WINTER_BRIEF]),
        result([produced_campaign], briefs=[produced]),
        fake_client(),
    )


async def test_grade_case_matching_brief_passes_every_field(fake_client):
    produced = WINTER_BRIEF.model_copy(
        update={"season": " aw31", "budget": Budget(amount=Decimal("8000.00"), currency="gbp")}
    )

    grades = await grade_brief(fake_client, produced)

    assert field_grades(grades) == [
        ("Loomhouse / Winter Knit Edit", field, True)
        for field in [
            "brand",
            "campaign_name",
            "season",
            "deadline",
            "budget",
            "mandatories",
            "references",
            "deliverables",
            *["deliverable_dimensions", "deliverable_due_date"] * 3,
        ]
    ]


async def test_grade_case_pairs_briefs_by_name_but_grades_name_case(fake_client):
    produced = WINTER_BRIEF.model_copy(
        update={"brand": "loomhouse", "campaign_name": " Winter Knit Edit "}
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["brand"]


async def test_grade_case_wrong_exact_fields_fail(fake_client):
    produced = WINTER_BRIEF.model_copy(
        update={"season": "SS32", "deadline": date(2031, 11, 16), "budget": None}
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["season", "deadline", "budget"]


async def test_grade_case_mandatories_and_references_in_another_order_pass(fake_client):
    produced = WINTER_BRIEF.model_copy(
        update={
            "mandatories": ["logo bottom right", "Show the wool fibre up close"],
            "references": ["Mood board v2", "AW30 lookbook"],
        }
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == []


async def test_grade_case_a_missing_mandatory_fails(fake_client):
    produced = WINTER_BRIEF.model_copy(update={"mandatories": ["Logo bottom right"]})

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["mandatories"]


async def test_grade_case_deliverables_in_another_order_pass(fake_client):
    produced = WINTER_BRIEF.model_copy(
        update={"deliverables": list(reversed(WINTER_BRIEF.deliverables))}
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == []


async def test_grade_case_a_missing_deliverable_fails(fake_client):
    produced = WINTER_BRIEF.model_copy(update={"deliverables": WINTER_BRIEF.deliverables[1:]})

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["deliverables"]


async def test_grade_case_wrong_dimensions_fail_only_the_dimensions_check(fake_client):
    email = WINTER_BRIEF.deliverables[2].model_copy(update={"format_or_dimensions": "600x300"})
    produced = WINTER_BRIEF.model_copy(
        update={"deliverables": [*WINTER_BRIEF.deliverables[:2], email]}
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["deliverable_dimensions"]


async def test_grade_case_wrong_due_date_fails_only_the_due_date_check(fake_client):
    email = WINTER_BRIEF.deliverables[2].model_copy(update={"due_date": None})
    produced = WINTER_BRIEF.model_copy(
        update={"deliverables": [*WINTER_BRIEF.deliverables[:2], email]}
    )

    grades = await grade_brief(fake_client, produced)

    assert failed_fields(grades) == ["deliverable_due_date"]


async def test_grade_case_with_no_campaigns_gives_only_outcome_and_count_grades(fake_client):
    produced = result(brief_escalation=escalation("not_a_brief"), briefs=[])

    grades = await grade_case(eval_case(escalation="not_a_brief"), produced, fake_client())

    assert [(g.grader, g.field) for g in grades.grades] == [
        ("outcome", "brief_escalation"),
        ("pairing", "campaign_count"),
    ]
    assert grades.passed


async def test_grade_case_same_deliverables_with_due_dates_in_another_order_pass(fake_client):
    expected = WINTER_BRIEF.model_copy(
        update={
            "deliverables": [
                Deliverable(asset_type="image", channel="instagram", due_date=date(2031, 11, 1)),
                Deliverable(asset_type="image", channel="instagram", due_date=date(2031, 11, 9)),
            ]
        }
    )
    produced = expected.model_copy(update={"deliverables": list(reversed(expected.deliverables))})

    grades = await grade_case(
        eval_case(campaigns=[WINTER], briefs=[expected]),
        result([unfinished("Loomhouse", "Winter Knit Edit")], briefs=[produced]),
        fake_client(),
    )

    assert failed_fields(grades) == []


async def test_grade_case_an_empty_budget_counts_as_no_budget(fake_client):
    expected = WINTER_BRIEF.model_copy(update={"budget": None})
    produced = WINTER_BRIEF.model_copy(update={"budget": Budget()})

    grades = await grade_case(
        eval_case(campaigns=[WINTER], briefs=[expected]),
        result([unfinished("Loomhouse", "Winter Knit Edit")], briefs=[produced]),
        fake_client(),
    )

    assert failed_fields(grades) == []


async def test_grade_case_an_unnamed_campaign_brief_gets_no_field_grades(fake_client):
    unnamed = StructuredBrief(brand="Loomhouse", objective="Launch knitwear")
    produced = result(brief_escalation=escalation("unnamed_campaign"), briefs=[unnamed])

    grades = await grade_case(eval_case(escalation="unnamed_campaign"), produced, fake_client())

    assert field_grades(grades) == []


async def test_grade_case_deliverables_with_wrong_dimensions_still_pair_on_due_date(fake_client):
    expected = WINTER_BRIEF.model_copy(
        update={
            "deliverables": [
                story("1080x1920", date(2031, 10, 1)),
                story("1080x1350", date(2031, 10, 15)),
            ]
        }
    )
    produced = expected.model_copy(
        update={
            "deliverables": [
                story("1200x628", date(2031, 10, 15)),
                story("1080x1080", date(2031, 10, 1)),
            ]
        }
    )

    grades = await grade_case(
        eval_case(campaigns=[WINTER], briefs=[expected]),
        result([unfinished("Loomhouse", "Winter Knit Edit")], briefs=[produced]),
        fake_client(),
    )

    assert failed_fields(grades) == ["deliverable_dimensions", "deliverable_dimensions"]


def story(dimensions: str, due: date) -> Deliverable:
    return Deliverable(
        asset_type="image", channel="instagram", format_or_dimensions=dimensions, due_date=due
    )


async def grade_deliverables(fake_client, expected, produced):
    return await grade_case(
        eval_case(
            campaigns=[WINTER],
            briefs=[WINTER_BRIEF.model_copy(update={"deliverables": expected})],
        ),
        result(
            [unfinished("Loomhouse", "Winter Knit Edit")],
            briefs=[WINTER_BRIEF.model_copy(update={"deliverables": produced})],
        ),
        fake_client(),
    )


async def test_grade_case_deliverable_grades_do_not_depend_on_order(fake_client):
    a = story("1080x1920", date(2031, 10, 1))
    b = story("1080x1920", date(2031, 10, 15))
    p = story("1080x1920", date(2031, 10, 20))
    q = story("1080x1350", date(2031, 10, 15))

    runs = [
        await grade_deliverables(fake_client, expected, produced)
        for expected in ([a, b], [b, a])
        for produced in ([p, q], [q, p])
    ]

    assert {tuple(failed_fields(g)) for g in runs} == {
        ("deliverable_due_date", "deliverable_dimensions")
    }


async def test_grade_case_deliverables_with_swapped_due_dates_fail_both_due_dates(fake_client):
    expected = [story("1080x1920", date(2031, 10, 1)), story("1080x1350", date(2031, 10, 15))]
    swapped = [story("1080x1920", date(2031, 10, 15)), story("1080x1350", date(2031, 10, 1))]

    for produced in (swapped, list(reversed(swapped))):
        grades = await grade_deliverables(fake_client, expected, produced)

        assert failed_fields(grades) == ["deliverable_due_date", "deliverable_due_date"]


async def test_grade_case_a_campaign_named_twice_pairs_both_without_field_grades(fake_client):
    duplicate = ExpectedCampaign(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        outcome="escalated",
        reason="duplicate_campaign",
    )
    escalated = CampaignOutcome(
        brand="Loomhouse",
        campaign_name="Winter Knit Edit",
        escalation=escalation("duplicate_campaign"),
    )

    grades = await grade_case(
        eval_case(campaigns=[duplicate, duplicate], briefs=[WINTER_BRIEF, WINTER_BRIEF]),
        result([escalated, escalated], briefs=[WINTER_BRIEF, WINTER_BRIEF]),
        fake_client(),
    )

    assert count_grade(grades).passed
    assert field_grades(grades) == []
    assert grades.passed
