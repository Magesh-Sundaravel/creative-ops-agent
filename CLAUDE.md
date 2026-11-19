# creative-ops-agent

Agentic system that takes a fashion/lifestyle creative brief from intake to agency handoff.
A coordinator agent delegates to specialist subagents, which act on a mock client stack
(DAM, project tracker, brand guidelines) exposed through our own MCP server.

## How we work

- Work on the current build step only, one ticket at a time. Stop after each and summarise the diff.
- Make the smallest change that works. No speculative abstractions, no unused config.
- If a requirement is ambiguous, ask before writing code.
- Add deps with `uv add` (never pip) and say why.
- Branches: `<type>/<issue-number>-<short-name>` (e.g. `feat/12-intake-agent`).

Per build step: `/grill-with-docs` → `/to-spec` → `/to-tickets` → per ticket:
`/implement` → `/code-review` → user review → `/commit-push-pr`.
Setup/config steps skip spec and tickets.

## Stack

Python 3.12, uv, Pydantic v2, pytest + pytest-asyncio, Ruff.
`anthropic` SDK for the hand-written agent loop; Claude Agent SDK for subagent orchestration;
`mcp` SDK for the mock server. FastAPI backend, Streamlit UI, Docker.

## Architecture

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

## Layout

```
src/creative_ops/  agents/, loop/, tools/, schemas/, guards/, api/, config.py
mcp_server/        mock DAM / tracker / brand guidelines + JSON seed data
ui/                streamlit_app.py
evals/             datasets/*.jsonl, run_evals.py, graders
tests/             unit + integration (mirrors src/)
docs/decisions/    ADRs
```

## Commands

```bash
uv sync
uv run pytest tests/unit -q
uv run ruff check . && uv run ruff format .
uv run python -m mcp_server
uv run uvicorn creative_ops.api.main:app --reload
uv run streamlit run ui/streamlit_app.py
uv run python evals/run_evals.py
```

## Build plan

0. [ ] Claude Code setup: rules, settings, hooks, skills  ← CURRENT
1. [ ] Scaffold: uv, Ruff, pytest, pre-commit, config, Pydantic schemas
2. [ ] Raw agent loop (`stop_reason`-driven), one dummy tool, unit tests
3. [ ] Intake agent: brief → `StructuredBrief` via tool use, `missing_fields`
4. [ ] Mock MCP server: DAM + tracker tools, brand guidelines as resources, seed data
5. [ ] Review + Handoff subagents, least-privilege tool sets
6. [ ] Coordinator: decomposition, parallel subagents, re-plan, escalation gates
7. [ ] Eval harness + first dataset (10–15 briefs)
8. [ ] FastAPI + Streamlit (submit brief, view trace, approve escalations)
9. [ ] CI: tests, Ruff, evals, `claude -p` review with structured output
10. [ ] Observability, Docker, README with architecture + eval results

Move the checkbox and CURRENT marker only when I confirm a step is done.

## Don't

- Add LangChain/LangGraph or any other agent framework.
- Use regex/text parsing to drive agent control flow.
- Let a subagent write to the tracker or DAM unless a gate allows it.
- Edit `.env`, expected outputs in `evals/datasets/`, or seed data without asking.
