from collections.abc import Sequence
from datetime import date

from creative_ops.agents.coordinator import (
    CampaignOutcome,
    CoordinatorResult,
    Escalation,
    EscalationReason,
)
from creative_ops.agents.handoff import HandoffPackage, RightsSummary
from creative_ops.loop.agent_loop import LoopResult, StopOutcome, TokenUsage
from evals.cases import EvalCase, ExpectedCampaign
from evals.graders import grade_case

LOOP = LoopResult(
    text="",
    outcome=StopOutcome.COMPLETED,
    messages=[],
    usage=TokenUsage(input_tokens=0, output_tokens=0),
)


def eval_case(
    escalation: EscalationReason | None = None, campaigns: Sequence[ExpectedCampaign] = ()
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case",
            "brief": "A Brief.",
            "today": "2031-10-01",
            "tags": [],
            "expected": {
                "briefs": [],
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


def result(
    campaigns: Sequence[CampaignOutcome] = (), brief_escalation: Escalation | None = None
) -> CoordinatorResult:
    return CoordinatorResult(
        campaigns=list(campaigns),
        escalation=brief_escalation,
        intakes=[],
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
