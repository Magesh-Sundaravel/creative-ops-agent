"""Streamlit UI: submit a Brief and see its campaign outcomes and open Escalations."""

import asyncio

import httpx
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

from creative_ops.agents.coordinator import CampaignOutcome, Escalation
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
    if brief.error:
        st.error(brief.error)
    if brief.results:
        result = brief.results[-1]
        for campaign in result.campaigns:
            outcome_card(campaign)
        if result.escalation:
            escalation_view(result.escalation, "Brief")
        for campaign in result.campaigns:
            if campaign.escalation:
                escalation_view(campaign.escalation, f"{campaign.brand} / {campaign.campaign_name}")
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
