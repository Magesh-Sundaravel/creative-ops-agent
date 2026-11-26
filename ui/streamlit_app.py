"""Streamlit UI: submit a Brief and see each stage's work, its outcomes and open Escalations."""

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
    EscalationReason,
    Retry,
)
from creative_ops.agents.review import Finding
from creative_ops.api.models import BriefDetail, BriefId, BriefSummary
from creative_ops.loop.agent_loop import LoopResult, TokenUsage
from creative_ops.schemas.brief import Budget

REASONS: dict[EscalationReason, str] = {
    "missing_critical_field": "A Critical Field is missing",
    "cannot_review": "The campaign can't be reviewed",
    "not_a_brief": "The Brief names no campaign",
    "unnamed_campaign": "The campaign has no name",
    "duplicate_campaign": "Two campaigns share a name",
    "unmatched_campaign": "A closed campaign may be listed under another name",
    "revisions_requested": "Review requested revisions",
    "run_failed": "An agent Run failed twice",
    "task_may_exist": "A Task may already exist for this campaign",
}
SOURCES = {"check": "Code check", "judgement": "Model judgement"}


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
        label = campaign_label(campaign.brand, campaign.campaign_name)
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


def campaign_label(brand: str | None, campaign_name: str | None) -> str:
    """A campaign's name for display, saying so when intake found no brand or name."""
    return f"{brand or 'Unknown brand'} / {campaign_name or 'Unnamed campaign'}"


def brief_label(item: BriefSummary) -> str:
    """A Brief's sidebar label: its campaigns, or when it came in if it has none yet."""
    names = ", ".join(campaign_label(c.brand, c.campaign_name) for c in item.campaigns)
    return f"{names or f'{item.submitted_at:%Y-%m-%d %H:%M}'} · {item.status}"


def _text(value: object) -> str:
    """A Structured Brief field's value as display text; empty when the Brief gives none."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, Budget):
        return " ".join(str(part) for part in (value.amount, value.currency) if part is not None)
    return str(value)


def findings_table(findings: list[Finding]) -> None:
    """Findings as a table of subject, rule, severity, explanation and source."""
    st.dataframe(
        [
            {
                "Subject": str(f.subject),
                "Rule": f.rule,
                "Severity": f.severity,
                "Explanation": f.explanation,
                "Source": SOURCES[f.source],
            }
            for f in findings
        ],
        hide_index=True,
    )


def escalation_details(escalation: Escalation) -> None:
    """An Escalation's readable reason, what it needs and the Brief's Ambiguities."""
    st.error(f"Escalated: {REASONS[escalation.reason]}")
    for need in escalation.needs:
        st.markdown(f"- Needs: {need}")
    for ambiguity in escalation.ambiguities:
        st.markdown(f"- Ambiguity: {ambiguity}")


def outcome_card(campaign: CampaignOutcome) -> None:
    """One campaign's outcome: handed off, revisions requested, escalated, or unfinished."""
    with st.container(border=True):
        st.markdown(f"**{campaign_label(campaign.brand, campaign.campaign_name)}**")
        if campaign.escalation:
            escalation_details(campaign.escalation)
        elif campaign.handoff:
            st.success(f"Handed off, Task {campaign.handoff.task_id}")
        elif campaign.revisions_requested:
            st.warning("Closed: revisions requested")
        elif campaign.task_may_exist:
            st.info(f"Closed against Task {campaign.task_id}")
        else:
            st.write("Unfinished")


def intake_view(results: list[CoordinatorResult]) -> None:
    """The latest Intake's Structured Briefs, with Missing Fields; critical ones stand out."""
    runs = [r for r in results if r.loop_result is not None]  # each Coordinator Run re-intakes
    intakes = runs[-1].intakes if runs else []
    campaigns = intakes[-1].campaigns if intakes else []
    if not campaigns:
        st.caption("No Structured Brief in the latest Run.")
    for c in campaigns:
        with st.container(border=True):
            st.markdown(f"**{campaign_label(c.brief.brand, c.brief.campaign_name)}**")
            critical = [f for f in c.missing_fields if f in c.critical_fields]
            other = [f for f in c.missing_fields if f not in c.critical_fields]
            if critical:
                st.error("Missing Critical Fields: " + ", ".join(f"`{f}`" for f in critical))
            if other:
                st.warning("Missing Fields: " + ", ".join(f"`{f}`" for f in other))
            st.dataframe(
                [
                    {"Field": name, "Value": _text(value)}
                    for name, value in c.brief
                    if name != "deliverables"
                ],
                hide_index=True,
            )
            if c.brief.deliverables:
                st.markdown("Deliverables")
                st.dataframe(
                    [d.model_dump(mode="json") for d in c.brief.deliverables], hide_index=True
                )


def review_view(campaigns: list[CampaignOutcome]) -> None:
    """Each campaign's Review: its Verdict and Findings."""
    reviewed = [(c, r) for c in campaigns if (r := c.review)]
    if not reviewed:
        st.caption("No Review yet.")
    for c, r in reviewed:
        with st.container(border=True):
            st.markdown(f"**{campaign_label(c.brand, c.campaign_name)}**")
            if r.verdict == "approved":
                st.success("Approved")
            else:
                st.warning("Revisions requested")
            if r.findings:
                findings_table(r.findings)
            else:
                st.caption("No Findings.")


def handoff_view(campaigns: list[CampaignOutcome]) -> None:
    """Each Handoff Package: deliverables and their Assets, legal lines, rights, notes, Task."""
    handed_off = [(c, p) for c in campaigns if (p := c.handoff)]
    if not handed_off:
        st.caption("No Handoff yet.")
    for c, p in handed_off:
        with st.container(border=True):
            st.markdown(f"**{campaign_label(c.brand, c.campaign_name)}**")
            st.success(f"Task {p.task_id}" + (f", due {p.deadline}" if p.deadline else ""))
            st.dataframe(
                [
                    {
                        **item.deliverable.model_dump(mode="json"),
                        "assets": ", ".join(f"{a.file_name} ({a.id})" for a in item.assets),
                        "usage_rights": " | ".join(
                            f"{a.id}: {', '.join(a.usage_rights.territories)}; "
                            f"{', '.join(a.usage_rights.channels)}; "
                            f"until {a.usage_rights.expires_on}"
                            for a in item.assets
                        ),
                    }
                    for item in p.deliverables
                ],
                hide_index=True,
            )
            st.markdown("**Legal lines**")
            for line in p.legal_lines:
                st.markdown(f"- {line}")
            st.markdown("**Production notes**")
            st.write(p.production_notes)
            if p.notes:
                st.markdown("**Advisory Findings**")
                findings_table(p.notes)


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
    """The selected Brief: its outcome and open Escalations, then each stage's work."""
    try:
        brief = get_brief(brief_id)
    except API_ERRORS as exc:
        st.error(f"Can't load the Brief: {exc}")  # polling goes on, and recovers with the API
        return
    st.markdown(f"**Status:** {brief.status}")
    if brief.error:
        st.error(brief.error)
    results = brief.results
    if results:
        st.header("Outcome")
        result = results[-1]
        open_escalations: list[tuple[Escalation, CampaignOutcome | None]] = []
        if result.escalation:
            with st.container(border=True):
                st.markdown("**The whole Brief**")
                escalation_details(result.escalation)
            open_escalations.append((result.escalation, None))
        for campaign in result.campaigns:
            outcome_card(campaign)
            if campaign.escalation:
                open_escalations.append((campaign.escalation, campaign))
        if brief.status == "escalated":
            answer_controls(brief_id, open_escalations, str(len(results)))
    st.header("Brief")
    st.text(brief.brief)
    for amendment in brief.amendments:
        st.markdown(f"**Amendment:** {amendment}")
    if results:
        st.header("Intake")
        intake_view(results)
        st.header("Review")
        review_view(results[-1].campaigns)
        st.header("Handoff")
        handoff_view(results[-1].campaigns)
        with st.expander("Trace"):
            trace_view(results)
    if brief.status != "running" and st.session_state.pop("polling", None) == brief_id:
        st.rerun()  # the Run finished: rerun the page so it stops polling


with st.sidebar:
    text = st.text_area("New Brief")
    if st.button("Submit"):
        if not text.strip():
            st.warning("Write a Brief before submitting.")
        else:
            try:
                st.session_state["selected"] = submit_brief(text)
            except API_ERRORS as exc:
                st.error(str(exc))
    st.divider()
    try:
        for item in list_briefs():
            if st.button(brief_label(item), key=item.id, width="stretch"):
                st.session_state["selected"] = item.id
    except API_ERRORS as exc:
        st.error(f"Can't reach the API: {exc}")

st.title("Creative Ops")
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
