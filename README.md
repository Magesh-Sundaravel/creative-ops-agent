# creative-ops-agent

An agentic system that takes a fashion/lifestyle creative brief from intake to agency handoff.
A coordinator agent delegates to specialist subagents, which act on a mock client stack
(DAM, project tracker, brand guidelines) exposed through our own MCP server.

> **Status:** early build. Scaffold is done; the raw agent loop is in progress.

## How it works

```
Streamlit UI ──> FastAPI ──> Coordinator agent
                                  ├── Intake   (brief -> StructuredBrief)
                                  ├── Review   (assets vs brief + brand rules)
                                  └── Handoff  (handoff package for production)
                                          │
                                   MCP server (mock client stack)
                                   ├── tools:     dam.search_assets, dam.get_asset,
                                   │              tracker.create_task, tracker.update_status
                                   └── resources: brand://guidelines/{brand}, brief://templates
```

Flow: raw brief → intake (validated `StructuredBrief`, gaps flagged) → human gate if critical
fields are missing → asset review → approve / request revisions → handoff package + tracker task.

## Stack

Python 3.12, uv, Pydantic v2, pytest + pytest-asyncio, Ruff. `anthropic` SDK for the
hand-written agent loop, Claude Agent SDK for subagent orchestration, `mcp` SDK for the mock
server, FastAPI backend, Streamlit UI, Docker. No LangChain/LangGraph.

## Getting started

```bash
uv sync
cp .env.example .env   # then set ANTHROPIC_API_KEY
uv run pytest tests/unit -q
uv run ruff check . && uv run ruff format .
```

Planned entry points (not built yet):

```bash
uv run python -m mcp_server
uv run uvicorn creative_ops.api.main:app --reload
uv run streamlit run ui/streamlit_app.py
uv run python evals/run_evals.py
```

## Layout

```
src/creative_ops/  agents/, loop/, tools/, schemas/, guards/, api/, config.py
mcp_server/        mock DAM / tracker / brand guidelines + JSON seed data
ui/                streamlit_app.py
evals/             datasets/*.jsonl, run_evals.py, graders
tests/             unit + integration (mirrors src/)
docs/decisions/    ADRs
```

## Build plan

- [x] 0. Claude Code setup
- [x] 1. Scaffold
- [ ] 2. Raw agent loop ← current
- [ ] 3. Intake agent
- [ ] 4. Mock MCP server
- [ ] 5. Review + Handoff subagents
- [ ] 6. Coordinator
- [ ] 7. Eval harness
- [ ] 8. FastAPI + Streamlit
- [ ] 9. CI
- [ ] 10. Observability, Docker, final README

Eval results and a fuller architecture write-up land in step 10.
