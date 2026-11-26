"""Coordinator: a Brief in, taken through intake, review and handoff (see ADR 0005)."""

from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Literal

from anthropic import AsyncAnthropic
from mcp import Client
from pydantic import BaseModel, ValidationError

from creative_ops.agents.handoff import (
    CannotHandoff,
    HandoffIncomplete,
    HandoffPackage,
    handoff,
)
from creative_ops.agents.intake import IntakeResult, NoValidSubmission, intake
from creative_ops.agents.review import CannotReview, NoValidReview, ReviewResult, review
from creative_ops.loop.agent_loop import (
    LoopResult,
    MaxIterationsExceeded,
    TokenUsage,
    UnexpectedStopReason,
    run,
)
from creative_ops.schemas.brief import StructuredBrief
from creative_ops.schemas.client_stack import Task, ToolError, permission_denied
from creative_ops.tools.tool import Tool, ToolResultError

SYSTEM_PROMPT = """\
You are the coordinator for a creative agency. Take the Brief from intake to handoff with \
your tools: call `intake` once to turn the Brief into one Structured Brief per campaign, \
then `review` each campaign, then `handoff` each campaign whose Review Verdict is approved. \
Name a campaign by its brand and campaign name exactly as `intake` returns them. A campaign \
marked `escalated` is waiting for a human, and one marked `handed_off`, `revisions_requested` or \
`task_may_exist` is done: don't review or hand any of them off. When there is nothing \
left to do, stop."""

MAX_ITERATIONS = 8

RunFailure = (
    MaxIterationsExceeded,
    UnexpectedStopReason,
    NoValidSubmission,
    NoValidReview,
    HandoffIncomplete,
)
"""A failed agent Run: retried once automatically, then escalated."""


class FailedRun(BaseModel):
    """The trace of one agent Run that failed."""

    agent: Literal["intake", "review", "handoff"]
    error: str
    """The failure's exception name."""
    messages: list[dict[str, Any]]
    usage: TokenUsage


def _failed_run(
    agent: Literal["intake", "review", "handoff"],
    exc: MaxIterationsExceeded
    | UnexpectedStopReason
    | NoValidSubmission
    | NoValidReview
    | HandoffIncomplete,
) -> FailedRun:
    """Record a failed Run from the trace its exception carries."""
    loop_failure = isinstance(exc, MaxIterationsExceeded | UnexpectedStopReason)
    trace = exc if loop_failure else exc.loop_result
    return FailedRun(
        agent=agent, error=type(exc).__name__, messages=list(trace.messages), usage=trace.usage
    )


class _NoteCreates:
    """A session that notes whether a Run's `tracker_create_task` may have created a Task."""

    def __init__(self, session: Client) -> None:
        """Wrap `session`; no create tried yet."""
        self._session = session
        self.task_may_exist = False
        self.task_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        """Everything but `call_tool` goes straight to the wrapped session."""
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        """Pass the call on; a create may have written unless its error proves otherwise."""
        if name != "tracker_create_task":
            return await self._session.call_tool(name, arguments, **kwargs)
        before = self.task_may_exist
        self.task_may_exist = True  # until the outcome is known
        result = await self._session.call_tool(name, arguments, **kwargs)
        if not result.is_error:
            self.task_id = Task.model_validate(result.structured_content).id
            return result
        # The same rule as Handoff's own slot: only a ToolError other than
        # upstream_unavailable proves this call wrote nothing.
        text = "".join(block.text for block in result.content if block.type == "text")
        try:
            written_nothing = ToolError.model_validate_json(text).type != "upstream_unavailable"
        except ValidationError:
            written_nothing = False
        if written_nothing:
            self.task_may_exist = before
        return result


class NoInput(BaseModel):
    """Input of a tool that takes none."""


class CampaignKey(BaseModel):
    """Names one campaign, as `intake` returned it."""

    brand: str
    campaign_name: str


class CampaignSummary(CampaignKey):
    """What the model sees of one campaign after intake."""

    missing_fields: list[str]
    escalated: bool
    handed_off: bool
    revisions_requested: bool
    task_may_exist: bool


class IntakeSummary(BaseModel):
    """What the model sees of an intake."""

    campaigns: list[CampaignSummary]
    ambiguities: list[str]


class ReviewSummary(BaseModel):
    """What the model sees of a review."""

    verdict: Literal["approved", "revisions_requested"]
    blocking_findings: int


class HandoffSummary(BaseModel):
    """What the model sees of a handoff."""

    task_id: str


EscalationReason = Literal[
    "missing_critical_field",
    "cannot_review",
    "not_a_brief",
    "unnamed_campaign",
    "duplicate_campaign",
    "unmatched_campaign",
    "revisions_requested",
    "run_failed",
    "task_may_exist",
]


class Escalation(BaseModel):
    """A decision handed to a human: why, the kind of answer it takes, and what it needs."""

    reason: EscalationReason
    answer: Literal["amend", "acknowledge", "retry", "confirm_task"]
    needs: list[str]
    ambiguities: list[str]


class CampaignOutcome(BaseModel):
    """What became of one campaign intake returned; no outcome means it is unfinished."""

    brand: str | None
    campaign_name: str | None
    review: ReviewResult | None = None
    """The campaign's latest Review in the Run that recorded this outcome."""
    handoff: HandoffPackage | None = None
    escalation: Escalation | None = None
    revisions_requested: bool = False
    """Closed as revisions requested, once a human acknowledged its Escalation."""
    task_may_exist: bool = False
    """Closed against a Task a failed Handoff may have created, once a human confirmed it."""
    task_id: str | None = None
    """The Task a failed Handoff created, when known."""

    @property
    def closed(self) -> bool:
        """Handed off, revisions requested, or a Task may exist: skipped after re-intake.

        Never re-run. A task-may-exist campaign reopens only when a human confirms no Task.
        """
        return (
            self.handoff is not None
            or self.revisions_requested
            or self.task_may_exist
            or (
                self.escalation is not None
                and self.escalation.reason in {"revisions_requested", "task_may_exist"}
            )
        )


class CoordinatorState(BaseModel):
    """Everything `resume()` needs to continue after an Escalation; serialisable."""

    brief: str
    escalation: Escalation | None
    """The Brief-level Escalation, if any."""
    campaigns: list[CampaignOutcome]

    @property
    def escalations(self) -> list[Escalation]:
        """Every open Escalation, Brief-level first."""
        return [
            e for e in [self.escalation, *(c.escalation for c in self.campaigns)] if e is not None
        ]


class CoordinatorResult(BaseModel):
    """Built by code from what the tools recorded, never from the model's final text.

    Every subagent `LoopResult` is in the recorded intakes, reviews and Handoff Packages,
    and every failed Run's trace in `failed_runs`.
    """

    campaigns: list[CampaignOutcome]
    escalation: Escalation | None
    """A Brief-level Escalation: no campaign could start."""
    intakes: list[IntakeResult]
    reviews: list[ReviewResult]
    failed_runs: list[FailedRun]
    loop_result: LoopResult | None
    """None when no Coordinator Run happened (an acknowledge, or a confirmed Task)."""
    state: CoordinatorState | None
    """Set when there is any Escalation, to pass to `resume()`."""


async def coordinate(
    client: AsyncAnthropic, session: Client, brief: str, today: date | None = None
) -> CoordinatorResult:
    """Take a Brief through intake, review and handoff."""
    return await _coordinate(client, session, brief, today, carried=[])


async def _coordinate(
    client: AsyncAnthropic,
    session: Client,
    brief: str,
    today: date | None,
    carried: list[CampaignOutcome],
) -> CoordinatorResult:
    """A Coordinator Run that skips the closed campaigns in `carried` and keeps their outcomes.

    Campaigns match `carried` by exact key only. If intake leaves out a closed campaign, any
    other campaign may be that one under another key, so every one not closed is escalated.
    """
    intakes: list[IntakeResult] = []
    reviews: list[ReviewResult] = []
    failed_runs: list[FailedRun] = []
    current_review: dict[tuple[str, str], ReviewResult] = {}
    intake_claimed = False
    reviewing: set[tuple[str, str]] = set()
    closed = {(o.brand, o.campaign_name) for o in carried}
    handoff_claimed = {(o.brand, o.campaign_name) for o in carried if o.handoff is not None}
    briefs: dict[tuple[str | None, str | None], StructuredBrief] = {}
    recorded = list(carried)
    """Every campaign's outcome, carried ones first; two campaigns may share a key."""
    outcomes: dict[tuple[str | None, str | None], CampaignOutcome] = {
        (o.brand, o.campaign_name): o for o in carried
    }
    brief_escalation: Escalation | None = None

    def campaign(key: CampaignKey) -> StructuredBrief:
        """The Structured Brief intake returned for this campaign."""
        found = briefs.get((key.brand, key.campaign_name))
        if found is None:
            raise ToolResultError(
                f"No campaign {key.brand!r} / {key.campaign_name!r}. "
                f"Use a brand and campaign name returned by intake: {sorted(map(str, briefs))}"
            )
        return found

    def escalation(reason: EscalationReason, needs: list[str]) -> Escalation:
        """An Escalation carrying the Brief's Ambiguities."""
        answers: dict[EscalationReason, Literal["acknowledge", "retry", "confirm_task"]] = {
            "revisions_requested": "acknowledge",
            "run_failed": "retry",
            "task_may_exist": "confirm_task",
        }
        return Escalation(
            reason=reason,
            answer=answers.get(reason, "amend"),
            needs=needs,
            ambiguities=intakes[-1].ambiguities if intakes else [],
        )

    async def twice[T](
        agent: Literal["intake", "review", "handoff"],
        start: Callable[[], Awaitable[T]],
        retry: Callable[[], bool] = lambda: True,
    ) -> T | FailedRun:
        """Start an agent Run, and once more if it fails and `retry()`; else the last failure."""
        for attempt in range(2):
            try:
                return await start()
            except RunFailure as exc:
                failed = _failed_run(agent, exc)
                failed_runs.append(failed)
                if attempt == 1 or not retry():
                    return failed
        raise AssertionError("unreachable")

    async def run_intake(_: NoInput) -> str:
        """Run Intake once per Coordinator Run, so briefs never change under review or handoff."""
        nonlocal intake_claimed, brief_escalation
        # Claim intake before the first await, so two calls in one response can't both run.
        if intake_claimed:
            raise ToolResultError(
                "Intake has already run, or is running, for this Brief. Use its campaigns, "
                "or stop if it failed."
            )
        intake_claimed = True
        result = await twice("intake", lambda: intake(client, brief, today))
        if isinstance(result, FailedRun):
            brief_escalation = escalation("run_failed", [f"Intake failed twice ({result.error})"])
            raise ToolResultError("Intake failed twice; the Brief is escalated to a human. Stop.")
        intakes.append(result)
        counts = Counter((c.brief.brand, c.brief.campaign_name) for c in result.campaigns)
        left_out = ", ".join(
            f"{o.brand!r} / {o.campaign_name!r}"
            for o in carried
            if (o.brand, o.campaign_name) not in counts
        )
        summaries: list[CampaignSummary] = []
        for c in result.campaigns:
            key = (c.brief.brand, c.brief.campaign_name)
            briefs[key] = c.brief
            # Closed before this Run (carried), so skipped, unless the key is now shared:
            # then each new campaign is recorded and escalated, and the closed one still gates.
            if key in closed and counts[key] == 1:
                outcome = outcomes[key]
            else:
                outcome = CampaignOutcome(brand=key[0], campaign_name=key[1])
                recorded.append(outcome)
                if key not in closed:
                    outcomes[key] = outcome
                if c.brief.campaign_name is None:
                    # Never guessed at: the name is how campaigns match across Intake runs.
                    outcome.escalation = escalation("unnamed_campaign", ["campaign_name"])
                elif counts[key] > 1:
                    needs = f"A different campaign name for each {key[0]!r} campaign {key[1]!r}"
                    if key in closed:
                        needs += "; one of them is already closed, so keep its name"
                    outcome.escalation = escalation("duplicate_campaign", [needs])
                elif left_out:
                    needs = (
                        f"Intake left out closed campaign(s) {left_out}, so {key[0]!r} campaign "
                        f"{key[1]!r} may be one of them under another name. Amend the Brief so "
                        "intake names each closed campaign exactly."
                    )
                    outcome.escalation = escalation("unmatched_campaign", [needs])
                elif c.critical_missing:
                    outcome.escalation = escalation("missing_critical_field", c.critical_fields)
            summaries.append(
                CampaignSummary(
                    brand=str(key[0]),
                    campaign_name=str(key[1]),
                    missing_fields=c.missing_fields,
                    escalated=outcome.escalation is not None,
                    handed_off=key in handoff_claimed,
                    revisions_requested=outcomes[key].revisions_requested,
                    task_may_exist=outcomes[key].task_may_exist,
                )
            )
        if not result.campaigns:
            brief_escalation = escalation("not_a_brief", ["A Brief naming at least one campaign"])
        return IntakeSummary(campaigns=summaries, ambiguities=result.ambiguities).model_dump_json()

    async def run_review(key: CampaignKey) -> str:
        """Run Review on one campaign and record its result, unless it is escalated (Gate)."""
        structured = campaign(key)
        if (key.brand, key.campaign_name) in handoff_claimed:
            raise ToolResultError(permission_denied("This campaign is already handed off."))
        escalated = outcomes[(key.brand, key.campaign_name)].escalation
        if escalated is not None:
            raise ToolResultError(
                permission_denied(f"This campaign is escalated to a human ({escalated.reason}).")
            )
        if outcomes[(key.brand, key.campaign_name)].closed:
            raise ToolResultError(permission_denied("This campaign is closed."))
        # Claim the key before the first await, so two calls in one response can't both run.
        if (key.brand, key.campaign_name) in reviewing:
            raise ToolResultError(permission_denied("This campaign is already being reviewed."))
        reviewing.add((key.brand, key.campaign_name))
        try:
            result = await twice("review", lambda: review(client, session, structured, today))
        except CannotReview as exc:
            outcomes[(key.brand, key.campaign_name)].escalation = escalation(
                "cannot_review", [str(exc)]
            )
            raise ToolResultError(permission_denied(f"Escalated to a human: {exc}")) from exc
        finally:
            reviewing.discard((key.brand, key.campaign_name))
        if isinstance(result, FailedRun):
            outcomes[(key.brand, key.campaign_name)].escalation = escalation(
                "run_failed", [f"Review failed twice ({result.error})"]
            )
            raise ToolResultError(permission_denied("Review failed twice; escalated to a human."))
        reviews.append(result)
        current_review[(key.brand, key.campaign_name)] = result
        outcomes[(key.brand, key.campaign_name)].review = result
        blocking = [f for f in result.findings if f.severity == "blocking"]
        if result.verdict == "revisions_requested":
            # Never retried: the work goes back to the creative team.
            outcomes[(key.brand, key.campaign_name)].escalation = escalation(
                "revisions_requested",
                [f"{f.subject} ({f.rule}): {f.explanation}" for f in blocking],
            )
        return ReviewSummary(
            verdict=result.verdict,
            blocking_findings=len(blocking),
        ).model_dump_json()

    async def run_handoff(key: CampaignKey) -> str:
        """Run Handoff on one reviewed campaign, at most once per Coordinator Run (Gate)."""
        structured = campaign(key)
        if (key.brand, key.campaign_name) in handoff_claimed:
            raise ToolResultError(
                permission_denied("This campaign is already handed off, or being handed off.")
            )
        if (key.brand, key.campaign_name) in reviewing:
            # Its Verdict may be about to change.
            raise ToolResultError(permission_denied("This campaign is being reviewed."))
        outcome = outcomes[(key.brand, key.campaign_name)]
        if outcome.escalation is not None or outcome.closed:
            raise ToolResultError(
                permission_denied("This campaign is escalated to a human, or closed.")
            )
        reviewed = current_review.get((key.brand, key.campaign_name))
        if reviewed is None:
            raise ToolResultError("Review this campaign before handing it off.")
        # Claim the key before the first await, so two calls in one response can't both run.
        handoff_claimed.add((key.brand, key.campaign_name))
        noted: list[_NoteCreates] = []

        def start() -> Awaitable[HandoffPackage]:
            noted.append(_NoteCreates(session))
            return handoff(client, noted[-1], structured, reviewed)

        try:
            # Retried only if the failed Run never tried to create a Task, so none can exist.
            package = await twice("handoff", start, retry=lambda: not noted[-1].task_may_exist)
        except CannotHandoff:
            # Refused before any Task was created.
            handoff_claimed.discard((key.brand, key.campaign_name))
            raise
        if isinstance(package, FailedRun):
            # Stays claimed: a Task may exist. Then never re-run, so the human closes it.
            outcome.task_id = noted[-1].task_id
            if noted[-1].task_id:
                outcome.escalation = escalation(
                    "task_may_exist",
                    [
                        f"Handoff failed ({package.error}) after creating tracker Task "
                        f"{noted[-1].task_id}. Check that Task, then confirm whether it exists."
                    ],
                )
            elif noted[-1].task_may_exist:
                outcome.escalation = escalation(
                    "task_may_exist",
                    [
                        f"Handoff failed ({package.error}) after trying to create a tracker "
                        "Task. Check the tracker for that Task, then confirm whether it exists."
                    ],
                )
            else:
                outcome.escalation = escalation(
                    "run_failed", [f"Handoff failed twice ({package.error})"]
                )
            raise ToolResultError(permission_denied("Handoff failed; escalated to a human."))
        outcomes[(key.brand, key.campaign_name)].handoff = package
        return HandoffSummary(task_id=package.task_id).model_dump_json()

    tools = [
        Tool(
            name="intake",
            description="Turn the Brief into one Structured Brief per campaign.",
            input_model=NoInput,
            handler=run_intake,
        ),
        Tool(
            name="review",
            description="Review one campaign's Assets. Returns its Review Verdict.",
            input_model=CampaignKey,
            handler=run_review,
        ),
        Tool(
            name="handoff",
            description="Hand one approved campaign to production. Returns the created Task.",
            input_model=CampaignKey,
            handler=run_handoff,
        ),
    ]
    loop_result = await run(
        client,
        f"<brief>\n{brief}\n</brief>",
        tools,
        system=SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        agent="coordinator",
    )
    state = CoordinatorState(brief=brief, escalation=brief_escalation, campaigns=recorded)
    return CoordinatorResult(
        campaigns=recorded,
        escalation=brief_escalation,
        intakes=intakes,
        reviews=reviews,
        failed_runs=failed_runs,
        loop_result=loop_result,
        state=state if state.escalations else None,
    )


class Amend(BaseModel):
    """A human's answer to an Escalation: text appended to the Brief, which Intake re-runs on."""

    kind: Literal["amend"] = "amend"
    text: str


class Acknowledge(BaseModel):
    """A human's answer to a revisions-requested Escalation: close the campaign, re-run nothing."""

    kind: Literal["acknowledge"] = "acknowledge"


class Retry(BaseModel):
    """A human's answer to a failed-Run Escalation: run the Coordinator again on the Brief."""

    kind: Literal["retry"] = "retry"


class ConfirmTask(CampaignKey):
    """A human's answer to one campaign's task-may-exist Escalation, after checking the tracker.

    A Task exists: close the campaign against it, re-run nothing. None exists: reopen the
    campaign and run the Coordinator again on the Brief, as Retry does.
    """

    kind: Literal["confirm_task"] = "confirm_task"
    task_exists: bool


class AnswerDoesNotFit(ValueError):
    """No open Escalation takes this answer: its kind, or the campaign it names. Nothing ran."""


def _names_open_task_may_exist(c: CampaignOutcome, answer: ConfirmTask) -> bool:
    """`c` is the campaign `answer` names, and its Escalation is task-may-exist."""
    return (
        c.escalation is not None
        and c.escalation.reason == "task_may_exist"
        and (c.brand, c.campaign_name) == (answer.brand, answer.campaign_name)
    )


def check_answer(
    state: CoordinatorState, answer: Amend | Acknowledge | Retry | ConfirmTask
) -> None:
    """Raise `AnswerDoesNotFit` unless `answer` fits an open Escalation; runs nothing."""
    if answer.kind not in {e.answer for e in state.escalations}:
        raise AnswerDoesNotFit(f"No open Escalation takes {answer.kind!r}.")
    if isinstance(answer, ConfirmTask) and not any(
        _names_open_task_may_exist(c, answer) for c in state.campaigns
    ):
        raise AnswerDoesNotFit(
            f"No task_may_exist Escalation for {answer.brand!r} / {answer.campaign_name!r}."
        )


async def resume(
    client: AsyncAnthropic,
    session: Client,
    state: CoordinatorState,
    answer: Amend | Acknowledge | Retry | ConfirmTask,
    today: date | None = None,
) -> CoordinatorResult:
    """Continue after an Escalation, with an answer that fits at least one open Escalation.

    Amend runs the Coordinator again on the amended Brief, and Retry on the same Brief, with
    fresh retry counts; closed campaigns are carried. Acknowledge closes every
    revisions-requested campaign without any Run. ConfirmTask answers the one campaign it
    names: a confirmed Task closes it without any Run, and no Task reopens it and runs the
    Coordinator again as Retry does, so every other unclosed campaign runs again too. Other
    Escalations stay open after an answer that runs nothing.
    """
    check_answer(state, answer)
    if isinstance(answer, Amend | Retry):
        brief = f"{state.brief}\n\n{answer.text}" if isinstance(answer, Amend) else state.brief
        carried = [c for c in state.campaigns if c.closed]
        return await _coordinate(client, session, brief, today, carried)
    if isinstance(answer, ConfirmTask):
        return await _confirm_task(client, session, state, answer, today)
    campaigns = [
        c.model_copy(update={"escalation": None, "revisions_requested": True})
        if c.escalation is not None and c.escalation.reason == "revisions_requested"
        else c
        for c in state.campaigns
    ]
    return _without_run(state, campaigns)


async def _confirm_task(
    client: AsyncAnthropic,
    session: Client,
    state: CoordinatorState,
    answer: ConfirmTask,
    today: date | None,
) -> CoordinatorResult:
    """Close the named task-may-exist campaign against its Task, or reopen it and run again."""

    campaigns = [
        c.model_copy(update={"escalation": None, "task_may_exist": answer.task_exists})
        if _names_open_task_may_exist(c, answer)
        else c
        for c in state.campaigns
    ]
    if not answer.task_exists:
        carried = [c for c in campaigns if c.closed]
        return await _coordinate(client, session, state.brief, today, carried)
    return _without_run(state, campaigns)


def _without_run(state: CoordinatorState, campaigns: list[CampaignOutcome]) -> CoordinatorResult:
    """The result of an answer that closed campaigns and ran nothing."""
    closed = state.model_copy(update={"campaigns": campaigns})
    return CoordinatorResult(
        campaigns=campaigns,
        escalation=state.escalation,
        intakes=[],
        reviews=[],
        failed_runs=[],
        loop_result=None,
        state=closed if closed.escalations else None,
    )
