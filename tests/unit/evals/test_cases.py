import pytest
from pydantic import ValidationError

from evals.cases import DATASETS_DIR, ExpectedCampaign, load_cases


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
