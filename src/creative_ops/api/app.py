"""The Brief API: submit a Brief, run the Coordinator in the background, read its outcome."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException

from creative_ops.agents.coordinator import CoordinatorResult
from creative_ops.api.models import (
    BriefDetail,
    BriefId,
    BriefRecord,
    BriefSummary,
    SubmitBrief,
)

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

    async def run(record: BriefRecord) -> None:
        """Run the Coordinator on a Brief, then record its result or its failure."""
        try:
            result = await coordinate(client, sessions[0], record.brief, date.today())
        except Exception as exc:
            logger.exception("Run failed for Brief %s", record.id)
            record.error = f"{type(exc).__name__}: {exc}"
            record.status = "failed"
            return
        record.results.append(result)
        record.state = result.state
        record.status = "escalated" if result.state else "done"

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

    app = FastAPI(lifespan=lifespan)

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
        task = asyncio.create_task(run(record))
        runs.add(task)
        task.add_done_callback(runs.discard)
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
