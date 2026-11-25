"""The Brief API: submit a Brief, run the Coordinator in the background, read its outcome."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from typing import Annotated, Any
from uuid import uuid4

from anthropic import AsyncAnthropic
from fastapi import Body, FastAPI, HTTPException

from creative_ops import tracing
from creative_ops.agents.coordinator import (
    Acknowledge,
    Amend,
    AnswerDoesNotFit,
    ConfirmTask,
    CoordinatorResult,
    Retry,
    check_answer,
)
from creative_ops.api.models import (
    BriefDetail,
    BriefId,
    BriefRecord,
    BriefSummary,
    SubmitBrief,
)
from creative_ops.logs import current_brief_id

logger = logging.getLogger(__name__)

Coordinate = Callable[[AsyncAnthropic, Any, str, date], Awaitable[CoordinatorResult]]


def create_app(
    coordinate: Coordinate,
    resume: Callable[..., Awaitable[CoordinatorResult]],
    open_session: Callable[[], AbstractAsyncContextManager[Any]],
    client: AsyncAnthropic,
) -> FastAPI:
    """Build the app; every Run shares the one client stack session its lifespan opens."""
    store: dict[str, BriefRecord] = {}
    runs: set[asyncio.Task[None]] = set()
    sessions: list[Any] = []

    async def run(record: BriefRecord, start: Callable[[], Awaitable[CoordinatorResult]]) -> None:
        """Await a Coordinator Run on a Brief, then record its result or its failure."""
        current_brief_id.set(record.id)  # this task's own context, so it never leaks to other Runs
        with tracing.trace_brief(record.id, record.brief) as trace:
            try:
                result = await start()
            except Exception as exc:
                logger.exception("Run failed for Brief %s", record.id)
                record.error = f"{type(exc).__name__}: {exc}"
                # A failed resume leaves the Escalation open to answer again.
                record.status = "failed" if record.state is None else "escalated"
                trace.update(output=record.error, level="ERROR", status_message=record.error)
                return
            record.error = None
            record.results.append(result)
            record.state = result.state
            record.status = "escalated" if result.state else "done"
            final = result.loop_result.text if result.loop_result else None
            trace.update(output=final or record.status)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Hold one client stack session open for the life of the app."""
        async with open_session() as session:
            sessions.append(session)
            try:
                yield
            finally:
                # Don't close the session under a Run still using it.
                for task in runs:
                    task.cancel()
                await asyncio.gather(*runs, return_exceptions=True)
                tracing.flush()

    app = FastAPI(lifespan=lifespan)

    def spawn(record: BriefRecord, start: Callable[[], Awaitable[CoordinatorResult]]) -> None:
        """Start a Run in the background, tracked until it finishes."""
        task = asyncio.create_task(run(record, start))
        runs.add(task)
        task.add_done_callback(runs.discard)

    @app.post("/briefs", status_code=202)
    async def submit(body: SubmitBrief) -> BriefId:
        """Start a Run on a new Brief in the background."""
        record = BriefRecord(
            id=uuid4().hex,
            status="running",
            submitted_at=datetime.now(UTC),
            brief=body.text,
            amendments=[],
            results=[],
            error=None,
        )
        store[record.id] = record
        spawn(record, lambda: coordinate(client, sessions[0], record.brief, date.today()))
        return BriefId(id=record.id)

    @app.post("/briefs/{brief_id}/answers", status_code=202)
    async def answer(
        brief_id: str,
        body: Annotated[Amend | Acknowledge | Retry | ConfirmTask, Body(discriminator="kind")],
    ) -> BriefId:
        """Answer an open Escalation and resume the Brief in the background."""
        record = store.get(brief_id)
        if record is None:
            raise HTTPException(status_code=404, detail="No such Brief.")
        if record.status != "escalated" or record.state is None:
            raise HTTPException(status_code=409, detail=f"The Brief is {record.status}.")
        if isinstance(body, Amend) and not body.text.strip():
            raise HTTPException(status_code=422, detail="An Amendment can't be empty.")
        try:
            check_answer(record.state, body)
        except AnswerDoesNotFit as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state = record.state

        async def resume_brief() -> CoordinatorResult:
            result = await resume(client, sessions[0], state, body, date.today())
            if isinstance(body, Amend):
                record.amendments.append(body.text)  # only once the amended Run has run
            return result

        record.status = "running"
        record.error = None
        spawn(record, resume_brief)
        return BriefId(id=record.id)

    @app.get("/briefs")
    async def list_briefs() -> list[BriefSummary]:
        """Every Brief, newest first."""
        newest_first = sorted(store.values(), key=lambda r: r.submitted_at, reverse=True)
        return [BriefSummary.model_validate(r, from_attributes=True) for r in newest_first]

    @app.get("/briefs/{brief_id}")
    async def get_brief(brief_id: str) -> BriefDetail:
        """One Brief and everything its Runs produced."""
        record = store.get(brief_id)
        if record is None:
            raise HTTPException(status_code=404, detail="No such Brief.")
        return BriefDetail.model_validate(record, from_attributes=True)

    return app
