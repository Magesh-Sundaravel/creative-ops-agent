# creative-ops-agent

A creative brief arrives as a messy email. Someone has to read it, pull out the fields, and
chase the client for whatever is missing before any work can start.

This agent does that first pass. Paste in a brief and get back structured fields, a list of
what the client left out, and a flag telling you when a person has to step in.

## What you get

Input, a brief as plain text:

```text
Acme spring launch. Two Instagram reels, due 15 Nov. Budget around £40k.
```

Output (abridged):

```text
campaigns:                        # one per campaign in the brief
  - brand:            Acme
    deliverables:     [{asset_type: reel, channel: Instagram, quantity: 2}]
    deadline:         2026-11-15  # year-less date, resolved from today
    budget:           40000 GBP
    missing_fields:   objective, target_audience, usage_rights, deliverables[0].format_or_dimensions, ...
    critical_missing: true        # no objective, so a human reviews before work starts
ambiguities:      ["Budget is stated as 'around £40k'"]
```

- Only what the brief states is filled in. Nothing is invented.
- Vague wording becomes an ambiguity for a person to confirm, not a guess.
- Missing fields are worked out by code, not by the model.

## Where it is

Intake works today. Next up is a mock client stack (asset library, project tracker, brand
guidelines), then agents that review assets against the brief and hand work off to production,
all run by a coordinator. See the [build plan](CLAUDE.md#build-plan).

## Try it

```bash
uv sync
cp .env.example .env   # then set ANTHROPIC_API_KEY
uv run pytest tests/unit -q
```

## Learn more

- [GLOSSARY.md](GLOSSARY.md): the terms this project uses (Brief, Missing Field, Critical Field)
- [docs/decisions/](docs/decisions/): why the agent loop is hand-written and how tool calls run
- [CLAUDE.md](CLAUDE.md): architecture, stack and how we work
