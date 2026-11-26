# creative-ops-agent

A coordinator agent that takes a fashion creative brief from intake to production handoff, delegating to intake, review and handoff subagents that act on a mock client stack.

<!-- TODO: demo (GIF or short video of the Streamlit UI: submit a brief, view the trace, answer an escalation) -->

## Problem

A creative brief arrives as a messy email. Before work can start, someone has to split it into campaigns, pull out the fields, chase the client for what is missing, check the delivered assets against the brief and the brand's rules, and open a task for production. Each step is mechanical but easy to get wrong, and a missed expiry date or a wrong image size costs a rework round.

## Examples

**1. A vague deadline is escalated, not guessed.** Today is 1 Oct 2031.

```text
Brief:   "... Deadline: before Black Friday. Budget: GBP 8,000."
Result:  campaign escalated (missing_critical_field)
         ambiguity: The deadline is given only as "before Black Friday", not as a date.
```

**2. A budget range is kept as a range.** The campaign still goes through to handoff.

```text
Brief:   "... Budget: somewhere between GBP 6,000 and 9,000."
Result:  handed_off
         ambiguity: The budget is given as a range: "between GBP 6,000 and 9,000".
```

**3. Assets that fail a rule come back as revisions.** Rights that expire before the deadline, or an image in the wrong dimensions, are found by code, not left to the model.

```text
Brief:   Winter Knit Edit, deadline 15 Oct 2032
Asset:   usage rights expire before 15 Oct 2032
Result:  revisions_requested (no tracker task is created)
```

<!-- TODO: confirm example 3 against the seed data's asset rights and quote the exact Finding text from a real run -->

## How it works

1. Intake turns the brief into one structured brief per campaign and lists missing fields, which code computes.
2. If a critical field is missing, the campaign is escalated to a person and goes no further.
3. Review compares the campaign's assets with the brief and the brand guidelines. Code checks the facts (status, channels, rights, dimensions); the model judges the rest.
4. Approved campaigns go to handoff, which writes the handoff package and creates one tracker task. Others come back as revisions.

The coordinator decides the order and which campaigns run in parallel. Code decides when to escalate.

## Results

One run of `evals/run_evals.py` on 26 Nov 2026, model `claude-haiku-5-5` for every agent and for the grader's judge calls.

| Measure | Value |
| --- | --- |
| Cases passed | 13 of 13 |
| Campaign outcomes | 27 of 27 |
| Fields | 166 of 166 |
| LLM judge checks | 68 of 68 |
| Tokens | 228,513 in, 35,339 out |
| Cost | $0.0405 |
| Wall-clock time | 76.6 s (3 cases at a time) |

Two earlier full runs on the same code each passed 12 of 13, failing a different case each time (`multi-campaign`, `wrong-dimensions`); both cases passed 4 of 4 when run alone.

The cases are written by the maintainer, and a single run is one sample. <!-- TODO: repeat the run to report variance -->

## Key decisions

- **Hand-written agent loop, no framework** ([ADR 0001](docs/decisions/0001-hand-written-agent-loop.md)). It decides from `stop_reason`, has a hard iteration cap, and keeps a complete message history per run, which traces and evals both use.
- **Code checks the facts, the model judges** ([ADR 0003](docs/decisions/0003-code-checks-facts-llm-judges.md)). The review model has no read tools, so it cannot drop or soften a deterministic finding.
- **Writes are gated at the tool boundary** ([ADR 0004](docs/decisions/0004-gates-at-the-tool-boundary.md)). The tracker write is refused unless the review was approved, and is allowed once per run.
- **The coordinator runs on the same loop** ([ADR 0005](docs/decisions/0005-coordinator-on-our-own-loop.md)), so there is one history format and one set of tests.
- **Langfuse Cloud, instrumented by hand** ([ADR 0006](docs/decisions/0006-langfuse-cloud-manual-instrumentation.md)), so spans follow the project's own run and iteration terms.

<!-- TODO: learnings, in the maintainer's words: what surprised you or went wrong while building this -->

## Limitations and next steps

- The client stack is a mock with seed data. No real DAM or tracker is connected.
- Briefs and traces go to Langfuse Cloud when keys are set, which is safe only because all data is mock (ADR 0006).
- The cost table is hand-kept and will go stale.
- Every agent runs on the fast model; there is no comparison across models. <!-- TODO: run the evals on a larger model and add the comparison -->
- The dataset has 13 cases, all written by the maintainer.

## Stack

Python 3.12, uv, Anthropic SDK, MCP SDK, Pydantic v2, FastAPI, Streamlit, Langfuse, pytest, Ruff, Docker.

<details>
<summary>Setup, environment variables and architecture</summary>

### Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
git clone https://github.com/Magesh-Sundaravel/creative-ops-agent.git
cd creative-ops-agent
uv sync
cp .env.example .env   # then set ANTHROPIC_API_KEY
uv run pytest tests/unit -q
```

Run the API and the UI in two terminals:

```bash
uv run uvicorn creative_ops.api.main:app --reload
uv run streamlit run ui/streamlit_app.py
```

Or with Docker, which serves the API on `127.0.0.1:8000` and the UI on `127.0.0.1:8501`:

```bash
docker compose up --build
```

Run the evals (calls the live API and costs a few cents):

```bash
uv run python evals/run_evals.py
```

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | yes | | API key |
| `MODEL_FAST` | no | `claude-haiku-5-5` | Model every agent and the eval judge use |
| `LOG_LEVEL` | no | `INFO` | Log level |
| `LANGFUSE_PUBLIC_KEY` | no | | With the secret key, turns on tracing |
| `LANGFUSE_SECRET_KEY` | no | | With the public key, turns on tracing |
| `LANGFUSE_HOST` | no | `https://cloud.langfuse.com` | `https://us.cloud.langfuse.com` for the US region |

### Architecture

```text
Streamlit UI ──> FastAPI ──> Coordinator agent
                                  ├── Intake   (brief -> one StructuredBrief per campaign)
                                  ├── Review   (assets vs brief + brand rules)
                                  └── Handoff  (handoff package for production)
                                          │
                                   MCP server (mock client stack)
                                   ├── tools:     dam_search_assets, dam_get_asset,
                                   │              tracker_create_task, tracker_update_status
                                   └── resources: brand://brands, brand://guidelines/{brand}
```

See [GLOSSARY.md](GLOSSARY.md) for the terms, [docs/decisions/](docs/decisions/) for the ADRs and [CLAUDE.md](CLAUDE.md) for the build plan.

</details>
