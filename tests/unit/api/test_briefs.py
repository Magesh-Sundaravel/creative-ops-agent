import threading
import time
from contextlib import asynccontextmanager
from datetime import date

import pytest
from fastapi.testclient import TestClient

from creative_ops.agents.coordinator import CoordinatorResult, CoordinatorState, Escalation
from creative_ops.api.app import create_app


def _result(escalated: bool = False) -> CoordinatorResult:
    escalation = Escalation(reason="not_a_brief", answer="amend", needs=["A Brief"], ambiguities=[])
    state = CoordinatorState(brief="b", escalation=escalation, campaigns=[]) if escalated else None
    return CoordinatorResult(
        campaigns=[],
        escalation=escalation if escalated else None,
        intakes=[],
        reviews=[],
        failed_runs=[],
        loop_result=None,
        state=state,
    )


class Fakes:
    """A fake `coordinate` that records its calls and returns or raises what the test set."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.result = _result()
        self.error: Exception | None = None
        self.gate: threading.Event | None = None

    async def coordinate(self, client, session, brief, today):
        import asyncio

        self.calls.append((client, session, brief, today))
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def http(fakes: Fakes):
    @asynccontextmanager
    async def open_session():
        yield "session"

    async def resume(*args, **kwargs):
        raise AssertionError("not used")

    app = create_app(fakes.coordinate, resume, open_session, "client")
    with TestClient(app) as client:
        yield client


def _wait(http: TestClient, brief_id: str, status: str) -> dict:
    for _ in range(100):
        body = http.get(f"/briefs/{brief_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"never {status}: {body}")


def test_submit_returns_202_and_id(http):
    response = http.post("/briefs", json={"text": "A brief"})

    assert response.status_code == 202
    assert set(response.json()) == {"id"}


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_422_and_starts_no_run(http, fakes, text):
    assert http.post("/briefs", json={"text": text}).status_code == 422
    assert fakes.calls == []


def test_running_then_done(http, fakes):
    fakes.gate = threading.Event()
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]

    running = http.get(f"/briefs/{brief_id}").json()
    assert running["status"] == "running"
    assert running["results"] == []

    fakes.gate.set()
    done = _wait(http, brief_id, "done")
    assert set(done) == {"id", "status", "submitted_at", "brief", "amendments", "results", "error"}
    assert done["brief"] == "A brief"
    assert done["amendments"] == []
    assert done["results"] == [fakes.result.model_dump(mode="json")]
    assert done["error"] is None


def test_escalated_when_result_has_state(http, fakes):
    fakes.result = _result(escalated=True)
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]

    assert _wait(http, brief_id, "escalated")["results"][0]["state"] is not None


def test_failed_when_coordinate_raises(http, fakes):
    fakes.error = RuntimeError("model down")
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]

    failed = _wait(http, brief_id, "failed")
    assert "RuntimeError" in failed["error"]
    assert "model down" in failed["error"]


def test_coordinate_gets_shared_session_and_today(http, fakes):
    first = http.post("/briefs", json={"text": "one"}).json()["id"]
    second = http.post("/briefs", json={"text": "two"}).json()["id"]
    _wait(http, first, "done")
    _wait(http, second, "done")

    assert {(c[0], c[1], c[3]) for c in fakes.calls} == {("client", "session", date.today())}
    assert sorted(c[2] for c in fakes.calls) == ["one", "two"]


def test_two_briefs_run_at_the_same_time(http, fakes):
    fakes.gate = threading.Event()
    ids = [http.post("/briefs", json={"text": t}).json()["id"] for t in ("one", "two")]

    for _ in range(100):
        if len(fakes.calls) == 2:
            break
        time.sleep(0.02)
    assert len(fakes.calls) == 2
    assert [http.get(f"/briefs/{i}").json()["status"] for i in ids] == ["running", "running"]
    fakes.gate.set()


def test_list_is_newest_first(http):
    first = http.post("/briefs", json={"text": "one"}).json()["id"]
    second = http.post("/briefs", json={"text": "two"}).json()["id"]

    listed = http.get("/briefs").json()

    assert [b["id"] for b in listed] == [second, first]
    assert set(listed[0]) == {"id", "status", "submitted_at"}


def test_unknown_id_is_404(http):
    assert http.get("/briefs/nope").status_code == 404
