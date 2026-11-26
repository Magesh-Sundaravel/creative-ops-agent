"""A Brief through the real API, MCP server and model. Run with `pytest -m integration`."""

import time
from datetime import date

import pytest
from anthropic import AsyncAnthropic
from fastapi.testclient import TestClient

from creative_ops.agents.coordinator import coordinate, resume
from creative_ops.api.app import create_app
from creative_ops.config import get_settings
from creative_ops.tools.mcp_adapter import open_client_stack

pytestmark = pytest.mark.integration

CLEAN_BRIEF = (
    "Loomhouse - Winter Knit Edit (AW31). Objective: launch the winter knitwear range to "
    "existing customers in the UK and EU. Audience: women aged 30-55 who buy natural-fibre "
    'clothing. Key message: "Warmth you can feel, made to last." Deliverables: 1 Instagram feed '
    "image (1080x1350) and 1 email header image (1200x600). Usage rights: Instagram and email, "
    "UK and EU, until 30 September 2032. Mandatory: show the wool fibre up close. Deadline: "
    "15 November 2031. Budget: GBP 8,000."
)
TODAY = date(2031, 10, 1)  # the eval cases' date, so the Brief's deadline stays in the future
NOT_A_BRIEF = (
    "Hi all, can we move Thursday's status call to 3pm? The client room is booked before then. "
    "Thanks!"
)


@pytest.fixture
def http():
    client = AsyncAnthropic(api_key=get_settings().anthropic_api_key.get_secret_value())

    async def coordinate_on_today(client, session, brief, _today):
        return await coordinate(client, session, brief, TODAY)

    app = create_app(coordinate_on_today, resume, open_client_stack, client)
    with TestClient(app) as http:
        yield http


def _finish(http: TestClient, text: str) -> dict:
    """Submit a Brief and wait until its Run stops running."""
    submitted = http.post("/briefs", json={"text": text})
    assert submitted.status_code == 202, submitted.text
    brief_id = submitted.json()["id"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = http.get(f"/briefs/{brief_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(1)
    raise AssertionError(f"still running after 600s: {brief_id}")


def test_submit_clean_brief_is_handed_off(http):
    body = _finish(http, CLEAN_BRIEF)

    assert body["status"] == "done", body["error"]
    [campaign] = body["results"][-1]["campaigns"]
    assert campaign["campaign_name"] == "Winter Knit Edit"
    assert campaign["handoff"] is not None


def test_submit_non_brief_is_escalated(http):
    body = _finish(http, NOT_A_BRIEF)

    assert body["status"] == "escalated", body["error"]
    assert body["results"][-1]["escalation"]["reason"] == "not_a_brief"
