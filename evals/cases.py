"""Eval Case: a Brief, the date it is run as of, and what it should produce."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator

from creative_ops.agents.coordinator import EscalationReason
from creative_ops.schemas.brief import StructuredBrief

DATASETS_DIR = Path(__file__).parent / "datasets"


class ExpectedCampaign(BaseModel):
    """What one campaign should end as; `reason` is set only when it is escalated."""

    brand: str
    campaign_name: str
    outcome: Literal["handed_off", "revisions_requested", "escalated"]
    reason: EscalationReason | None = None

    @model_validator(mode="after")
    def _reason_only_when_escalated(self) -> Self:
        """An escalated campaign names its reason; any other outcome has none."""
        if (self.outcome == "escalated") != (self.reason is not None):
            raise ValueError("reason is required when, and only when, outcome is 'escalated'")
        if self.reason == "revisions_requested":
            raise ValueError("use outcome 'revisions_requested', not that escalation reason")
        return self


class Expected(BaseModel):
    """The Structured Briefs, Ambiguities and outcome an Eval Case's Brief should produce."""

    briefs: list[StructuredBrief]
    ambiguities: list[str]
    escalation: EscalationReason | None
    """The Brief-level Escalation reason, or None."""
    campaigns: list[ExpectedCampaign]


class EvalCase(BaseModel):
    """One Brief with the date it is run as of and what it should produce."""

    id: str
    brief: str
    today: date
    tags: list[str]
    expected: Expected


def load_cases(datasets_dir: Path = DATASETS_DIR) -> list[EvalCase]:
    """Every Eval Case in the dataset files, one per non-blank JSONL line, in file order."""
    return [
        EvalCase.model_validate_json(line)
        for path in sorted(datasets_dir.glob("*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
