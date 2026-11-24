import asyncio
import threading
import time
from contextlib import asynccontextmanager
from datetime import date

import pytest
from fastapi.testclient import TestClient

from creative_ops.agents.coordinator import (
    CampaignOutcome,
    CoordinatorResult,
    CoordinatorState,
    Escalation,
)
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
        self.resumes: list[tuple] = []
        self.resumed = _result()
        self.resume_gate: threading.Event | None = None
        self.resume_error: Exception | None = None

    async def resume(self, client, session, state, answer, today):
        self.resumes.append((client, session, state, answer, today))
        if self.resume_gate is not None:
            await asyncio.to_thread(self.resume_gate.wait)
        if self.resume_error:
            raise self.resume_error
        return self.resumed

    async def coordinate(self, client, session, brief, today):
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

    app = create_app(fakes.coordinate, fakes.resume, open_session, "client")
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


def _escalated(http, fakes, result: CoordinatorResult | None = None) -> str:
    fakes.result = result or _result(escalated=True)
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]
    _wait(http, brief_id, "escalated")
    return brief_id


def _task_may_exist_result() -> CoordinatorResult:
    escalation = Escalation(
        reason="task_may_exist", answer="confirm_task", needs=["Check"], ambiguities=[]
    )
    campaign = CampaignOutcome(brand="B", campaign_name="C", escalation=escalation)
    state = CoordinatorState(brief="b", escalation=None, campaigns=[campaign])
    return CoordinatorResult(
        campaigns=[campaign],
        escalation=None,
        intakes=[],
        reviews=[],
        failed_runs=[],
        loop_result=None,
        state=state,
    )


def test_amend_resumes_and_appends_the_amendment(http, fakes):
    brief_id = _escalated(http, fakes)
    stored = fakes.result.state

    response = http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "More"})

    assert response.status_code == 202
    body = _wait(http, brief_id, "done")
    assert body["amendments"] == ["More"]
    assert body["brief"] == "A brief"
    assert len(body["results"]) == 2
    _, session, state, answer, today = fakes.resumes[0]
    assert (session, state, answer.kind, today) == ("session", stored, "amend", date.today())


def test_resume_result_sets_escalated_again(http, fakes):
    brief_id = _escalated(http, fakes)
    fakes.resumed = _result(escalated=True)

    http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "More"})

    body = _wait(http, brief_id, "escalated")
    assert len(body["results"]) == 2


def test_resume_raising_marks_failed(http, fakes):
    brief_id = _escalated(http, fakes)
    fakes.resume_error = RuntimeError("down")

    http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "More"})

    body = _wait(http, brief_id, "failed")
    assert body["error"] == "RuntimeError: down"
    assert len(body["results"]) == 1


def test_answer_while_running_is_409(http, fakes):
    fakes.gate = threading.Event()
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]

    response = http.post(f"/briefs/{brief_id}/answers", json={"kind": "retry"})

    assert response.status_code == 409
    fakes.gate.set()


def test_answer_while_resuming_is_409(http, fakes):
    brief_id = _escalated(http, fakes)
    fakes.resume_gate = threading.Event()
    http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "More"})

    response = http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "Again"})

    assert response.status_code == 409
    fakes.resume_gate.set()


def test_answer_when_done_is_409(http, fakes):
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]
    _wait(http, brief_id, "done")

    assert http.post(f"/briefs/{brief_id}/answers", json={"kind": "retry"}).status_code == 409


def test_answer_when_failed_is_409(http, fakes):
    fakes.error = RuntimeError("down")
    brief_id = http.post("/briefs", json={"text": "A brief"}).json()["id"]
    _wait(http, brief_id, "failed")

    assert http.post(f"/briefs/{brief_id}/answers", json={"kind": "retry"}).status_code == 409


def test_answer_to_unknown_id_is_404(http):
    assert http.post("/briefs/nope/answers", json={"kind": "retry"}).status_code == 404


def test_answer_no_escalation_takes_is_422_and_runs_nothing(http, fakes):
    brief_id = _escalated(http, fakes)

    response = http.post(f"/briefs/{brief_id}/answers", json={"kind": "retry"})

    assert response.status_code == 422
    assert "retry" in response.json()["detail"]
    assert fakes.resumes == []
    assert http.get(f"/briefs/{brief_id}").json()["status"] == "escalated"


def test_rejected_amend_is_not_recorded(http, fakes):
    brief_id = _escalated(http, fakes, _task_may_exist_result())

    http.post(f"/briefs/{brief_id}/answers", json={"kind": "amend", "text": "x"})

    assert http.get(f"/briefs/{brief_id}").json()["amendments"] == []


def test_confirm_task_naming_no_such_campaign_is_422(http, fakes):
    brief_id = _escalated(http, fakes, _task_may_exist_result())
    answer = {"kind": "confirm_task", "brand": "B", "campaign_name": "Other", "task_exists": True}

    assert http.post(f"/briefs/{brief_id}/answers", json=answer).status_code == 422
    assert fakes.resumes == []


def test_confirm_task_naming_the_campaign_resumes(http, fakes):
    brief_id = _escalated(http, fakes, _task_may_exist_result())
    answer = {"kind": "confirm_task", "brand": "B", "campaign_name": "C", "task_exists": True}

    r = http.post(f"/briefs/{brief_id}/answers", json=answer)
    assert r.status_code == 202, r.text
    _wait(http, brief_id, "done")
    assert fakes.resumes[0][3].task_exists is True


def test_unknown_answer_kind_is_422(http, fakes):
    brief_id = _escalated(http, fakes)

    assert http.post(f"/briefs/{brief_id}/answers", json={"kind": "nope"}).status_code == 422
