import pytest
from pydantic import ValidationError

from evals.cases import DATASETS_DIR, Expected, ExpectedCampaign, load_cases


def test_load_cases_every_dataset_line_parses_with_a_unique_id():
    files = sorted(DATASETS_DIR.glob("*.jsonl"))
    lines = [line for f in files for line in f.read_text().splitlines() if line.strip()]

    cases = load_cases()

    assert files
    assert len(cases) == len(lines)
    assert len({c.id for c in cases}) == len(cases)


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("escalated", None),
        ("handed_off", "cannot_review"),
        ("escalated", "revisions_requested"),
    ],
    ids=["escalated-without-reason", "reason-without-escalation", "escalated-for-revisions"],
)
def test_expected_campaign_outcome_and_reason_that_cannot_match_a_result_is_rejected(
    outcome, reason
):
    with pytest.raises(ValidationError):
        ExpectedCampaign(brand="b", campaign_name="c", outcome=outcome, reason=reason)


@pytest.mark.parametrize(
    ("brief_names", "campaign_names"),
    [
        (["Resort"], ["Winter"]),
        (["Winter", "Resort"], ["Winter"]),
        ([], ["Winter"]),
        (["Winter", " winter "], ["Winter"]),
    ],
    ids=["mismatched", "brief-without-campaign", "campaign-without-brief", "brief-twice"],
)
def test_expected_briefs_and_campaigns_naming_different_campaigns_are_rejected(
    brief_names, campaign_names
):
    with pytest.raises(ValidationError, match="briefs and campaigns differ"):
        Expected.model_validate(
            {
                "briefs": [{"brand": "Loomhouse", "campaign_name": n} for n in brief_names],
                "ambiguities": [],
                "escalation": None,
                "campaigns": [
                    {"brand": "Loomhouse", "campaign_name": n, "outcome": "handed_off"}
                    for n in campaign_names
                ],
            }
        )


def test_expected_brief_and_campaign_differing_only_in_case_and_spaces_are_accepted():
    expected = Expected.model_validate(
        {
            "briefs": [{"brand": " loomhouse", "campaign_name": " winter "}],
            "ambiguities": [],
            "escalation": None,
            "campaigns": [
                {"brand": "Loomhouse", "campaign_name": "Winter", "outcome": "handed_off"}
            ],
        }
    )

    assert len(expected.campaigns) == 1
