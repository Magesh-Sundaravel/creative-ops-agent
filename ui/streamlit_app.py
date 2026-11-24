"""Streamlit UI: submit a Brief and see its campaign outcomes and open Escalations."""

import asyncio

import httpx
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

from creative_ops.agents.coordinator import (
    Acknowledge,
    Amend,
    CampaignOutcome,
    ConfirmTask,
    Escalation,
    Retry,
)
from creative_ops.api.models import BriefDetail, BriefId, BriefSummary


class UISettings(BaseSettings):
    """The UI needs only where the API is; it never holds the Anthropic key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_url: str = "http://localhost:8000"


API = UISettings().api_url
API_ERRORS = (httpx.HTTPError, ValueError)
"""A failed request, or a body that isn't the shape the API promises."""

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
    brief_id: str, escalation: Escalation, key: str, campaign: CampaignOutcome | None
) -> None:
    """The control that matches the kind of answer the Escalation takes."""
    if escalation.answer == "amend":
        text = st.text_area("Amendment", key=f"amend-{key}")
        if st.button("Amend", key=f"amend-button-{key}", disabled=not text.strip()):
            _send(brief_id, Amend(text=text))
    elif escalation.answer == "confirm_task" and campaign:
        exists = st.radio("Task exists?", ["Yes", "No"], key=f"confirm-{key}", horizontal=True)
        if st.button("Confirm", key=f"confirm-button-{key}"):
            _send(
                brief_id,
                ConfirmTask(
                    brand=campaign.brand or "",
                    campaign_name=campaign.campaign_name or "",
                    task_exists=exists == "Yes",
                ),
            )
    elif escalation.answer != "confirm_task" and st.button(
        escalation.answer.capitalize(), key=f"{escalation.answer}-{key}"
    ):
        _send(brief_id, Acknowledge() if escalation.answer == "acknowledge" else Retry())


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
        if result.escalation:
            escalation_view(result.escalation, "Brief")
            if brief.status == "escalated":
                answer_controls(brief_id, result.escalation, f"brief-{len(brief.results)}", None)
        for i, campaign in enumerate(result.campaigns):
            if campaign.escalation:
                escalation_view(campaign.escalation, f"{campaign.brand} / {campaign.campaign_name}")
                if brief.status == "escalated":
                    answer_controls(
                        brief_id, campaign.escalation, f"{i}-{len(brief.results)}", campaign
                    )
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
