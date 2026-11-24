"""Request and response models of the Brief API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from creative_ops.agents.coordinator import CoordinatorResult, CoordinatorState

Status = Literal["running", "escalated", "done", "failed"]


class SubmitBrief(BaseModel):
    """A Brief as the client sent it."""

    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, text: str) -> str:
        """Reject a Brief with nothing in it, keeping the text exactly as received."""
        if not text.strip():
            raise ValueError("A Brief can't be empty.")
        return text


class BriefId(BaseModel):
    id: str


class BriefSummary(BriefId):
    status: Status
    submitted_at: datetime


class BriefDetail(BriefSummary):
    brief: str
    amendments: list[str]
    results: list[CoordinatorResult]
    error: str | None


class BriefRecord(BriefDetail):
    """What the store keeps for a Brief: its detail and the state `resume` needs."""

    state: CoordinatorState | None = None
