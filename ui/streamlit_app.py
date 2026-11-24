"""Streamlit UI: submit a Brief and see its campaign outcomes and open Escalations."""

import asyncio
from typing import Any

import httpx
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

from creative_ops.agents.coordinator import (
    Acknowledge,
    Amend,
    CampaignOutcome,
    ConfirmTask,
    CoordinatorResult,
    Escalation,
    Retry,
)
from creative_ops.api.models import BriefDetail, BriefId, BriefSummary
from creative_ops.loop.agent_loop import LoopResult, TokenUsage


class UISettings(BaseSettings):
    """The UI needs only where the API is; it never holds the Anthropic key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_url: str = "http://localhost:8000"


API = UISettings().api_url
# A failed request, or a body that isn't the shape the API promises.
API_ERRORS = (httpx.HTTPError, ValueError)

st.set_page_config(page_title="Creative Ops", layout="wide")


async def _request(method: str, path: str, **kwargs) -> object:
    """One API call; the JSON body of a successful response."""
    async with httpx.AsyncClient(base_url=API, timeout=10) as client:
        response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def get_brief(brief_id: str) -> BriefDetail:
    """One Brief from the API."""
    return BriefDetail.model_validate(asyncio.run(_request("GET", f"/briefs/{brief_id}")))


def list_briefs() -> list[BriefSummary]:
    """Every Brief from the API, newest first."""
    return [BriefSummary.model_validate(item) for item in asyncio.run(_request("GET", "/briefs"))]


def submit_brief(text: str) -> str:
    """Submit a Brief; the id the API gave it."""
    return BriefId.model_validate(asyncio.run(_request("POST", "/briefs", json={"text": text}))).id


def post_answer(brief_id: str, answer: Amend | Acknowledge | Retry | ConfirmTask) -> None:
    """Answer an open Escalation; the Brief resumes in the background."""
    asyncio.run(_request("POST", f"/briefs/{brief_id}/answers", json=answer.model_dump()))


def _send(brief_id: str, answer: Amend | Acknowledge | Retry | ConfirmTask) -> None:
    """Post an answer, showing the API's message when it refuses."""
    try:
        post_answer(brief_id, answer)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()["detail"]
            if isinstance(detail, list):  # FastAPI's validation errors
                detail = "; ".join(str(item["msg"]) for item in detail)
        except (ValueError, KeyError, TypeError):
            detail = exc.response.text or str(exc)
        st.error(str(detail))
        if exc.response.status_code == 409:
            st.session_state["polling"] = brief_id
            st.rerun()  # stale view: refetch the Brief
    except API_ERRORS as exc:
        st.error(str(exc))
    else:
        st.session_state["polling"] = brief_id
        st.rerun()  # the Brief is running again: rerun the page so it polls


def answer_controls(
    brief_id: str, escalations: list[tuple[Escalation, CampaignOutcome | None]], key: str
) -> None:
    """One control per kind of answer the open Escalations take; ConfirmTask is per campaign."""
    kinds = {e.answer for e, _ in escalations}
    if "amend" in kinds:
        text = st.text_area("Amendment (applies to the whole Brief)", key=f"amend-{key}")
        if st.button("Amend", key=f"amend-button-{key}", disabled=not text.strip()):
            _send(brief_id, Amend(text=text))
    if "acknowledge" in kinds and st.button("Acknowledge all", key=f"acknowledge-{key}"):
        _send(brief_id, Acknowledge())
    if "retry" in kinds and st.button("Retry the Brief", key=f"retry-{key}"):
        _send(brief_id, Retry())
    for i, (escalation, campaign) in enumerate(escalations):
        if escalation.answer != "confirm_task":
            continue
        if campaign is None or campaign.brand is None or campaign.campaign_name is None:
            st.warning("This Escalation needs a Task check, but names no campaign to answer.")
            continue
        label = f"{campaign.brand} / {campaign.campaign_name}"
        exists = st.radio(f"Task exists for {label}?", ["Yes", "No"], key=f"confirm-{i}-{key}")
        if st.button("Confirm", key=f"confirm-button-{i}-{key}"):
            _send(
                brief_id,
                ConfirmTask(
                    brand=campaign.brand,
                    campaign_name=campaign.campaign_name,
                    task_exists=exists == "Yes",
                ),
            )


def outcome_card(campaign: CampaignOutcome) -> None:
    """One campaign's outcome: handed off, revisions requested, escalated, or unfinished."""
    with st.container(border=True):
        st.subheader(f"{campaign.brand} / {campaign.campaign_name}")
        if campaign.handoff:
            st.success(f"Handed off, Task {campaign.handoff.task_id}")
        elif campaign.revisions_requested:
            st.warning("Revisions requested")
        elif campaign.task_may_exist:
            st.info(f"Closed against Task {campaign.task_id}")
        elif campaign.escalation:
            st.error(f"Escalated: {campaign.escalation.reason}")
        else:
            st.write("Unfinished")


def escalation_view(escalation: Escalation, title: str) -> None:
    """An open Escalation's reason, what it needs and the Brief's Ambiguities."""
    with st.container(border=True):
        st.markdown(f"**{title}**: {escalation.reason}")
        for need in escalation.needs:
            st.markdown(f"- Needs: {need}")
        for ambiguity in escalation.ambiguities:
            st.markdown(f"- Ambiguity: {ambiguity}")


def trace_runs(
    results: list[CoordinatorResult],
) -> list[tuple[str, list[dict[str, Any]], TokenUsage, str | None]]:
    """Every Run across the results in order, as (agent, messages, usage, error).

    A Handoff carried into a later result is shown once.
    """
    runs: list[tuple[str, list[dict[str, Any]], TokenUsage, str | None]] = []
    seen: set[str] = set()

    def add(agent: str, run: LoopResult) -> None:
        key = run.model_dump_json()
        if key not in seen:
            seen.add(key)
            runs.append((agent, run.messages, run.usage, None))

    for result in results:
        if result.loop_result:
            add("coordinator", result.loop_result)
        for intake in result.intakes:
            add("intake", intake.loop_result)
        for review in result.reviews:
            add("review", review.loop_result)
        for campaign in result.campaigns:
            if campaign.handoff:
                add("handoff", campaign.handoff.loop_result)
        for failed in result.failed_runs:
            runs.append((f"{failed.agent} (failed)", failed.messages, failed.usage, failed.error))
    return runs


def trace_view(results: list[CoordinatorResult]) -> None:
    """The Trace: one section per Run with its token usage and its messages as raw JSON."""
    for n, (agent, messages, usage, error) in enumerate(trace_runs(results), start=1):
        st.markdown(f"**{n}. {agent}**")
        if error:
            st.error(error)
        st.caption(f"{usage.input_tokens} input tokens, {usage.output_tokens} output tokens")
        st.json(messages, expanded=False)


def brief_pane(brief_id: str) -> None:
    """The selected Brief's status, campaign outcomes and open Escalations."""
    try:
        brief = get_brief(brief_id)
    except API_ERRORS as exc:
        st.error(f"Can't load the Brief: {exc}")  # polling goes on, and recovers with the API
        return
    st.markdown(f"**Status:** {brief.status}")
    with st.expander("Brief as received"):
        st.text(brief.brief)
    if brief.error:
        st.error(brief.error)
    if brief.results:
        result = brief.results[-1]
        for campaign in result.campaigns:
            outcome_card(campaign)
        open_escalations: list[tuple[Escalation, CampaignOutcome | None]] = []
        if result.escalation:
            escalation_view(result.escalation, "Brief")
            open_escalations.append((result.escalation, None))
        for campaign in result.campaigns:
            if campaign.escalation:
                escalation_view(campaign.escalation, f"{campaign.brand} / {campaign.campaign_name}")
                open_escalations.append((campaign.escalation, campaign))
        if brief.status == "escalated":
            answer_controls(brief_id, open_escalations, str(len(brief.results)))
    if brief.results:
        with st.expander("Trace"):
            trace_view(brief.results)
    if brief.amendments:
        st.markdown("**Amendments**")
        for amendment in brief.amendments:
            st.markdown(f"> {amendment}")
    if brief.status != "running" and st.session_state.pop("polling", None) == brief_id:
        st.rerun()  # the Run finished: rerun the page so it stops polling


with st.sidebar:
    text = st.text_area("New Brief")
    if st.button("Submit"):
        try:
            st.session_state["selected"] = submit_brief(text)
        except API_ERRORS as exc:
            st.error(str(exc))
    st.divider()
    try:
        for item in list_briefs():
            label = f"{item.submitted_at:%Y-%m-%d %H:%M:%S} · {item.status}"
            if st.button(label, key=item.id):
                st.session_state["selected"] = item.id
    except API_ERRORS as exc:
        st.error(f"Can't reach the API: {exc}")

selected = st.session_state.get("selected")
if selected:
    try:
        running = get_brief(selected).status == "running"
    except API_ERRORS as exc:
        st.error(f"Can't load the Brief: {exc}")
    else:
        if running:
            st.session_state["polling"] = selected
        # Poll every 2 seconds only while the Brief is running.
        st.fragment(run_every=2 if running else None)(brief_pane)(selected)
else:
    st.write("Submit a Brief, or select one from the sidebar.")
