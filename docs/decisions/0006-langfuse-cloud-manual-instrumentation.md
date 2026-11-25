# Observability is Langfuse Cloud, instrumented by hand, with our own cost table

Traces and per-Run metrics go to Langfuse Cloud through its SDK, called from our own loop and API. One Langfuse trace is one Brief's Trace: each Run is a span tagged with its agent, each Iteration is a generation (model, tokens, cost, latency, outcome), and each tool call is a span under the Iteration that requested it. Failed Runs are recorded at error level. The API and the evals runner both trace, and an eval case is one trace tagged `eval`. Logs stay plain text with the Brief id on every line, carried by a `contextvar`, which the loop also uses to find the current trace without a new parameter.

## Considered Options

- **Stdlib JSON logs, no backend.** Simplest, but nothing to query or look at, and no cost or latency view.
- **Self-hosted Langfuse.** Adds Postgres, ClickHouse, Redis and MinIO to the compose file for a demo project.
- **Auto-instrumentation of the Anthropic SDK.** Less code, but spans would not follow our Run, Iteration and Trace vocabulary, and tool calls and Gates would be invisible.
- **Langfuse's own model price definitions.** We compute cost ourselves from the official Haiku 5.5 rates (tiered at 100k prompt tokens) and send it as `cost_details`, so the price is visible in code and in the logs. Every agent runs on the fast model, so this is the only entry. An unknown model gets tokens and no cost.

## Consequences

Langfuse keys are optional settings. With none set, tracing is a no-op and tests and CI need no Langfuse secrets; an outage never fails a Run.

Full prompts, responses and tool inputs and outputs are sent to Langfuse Cloud. That is safe only because every Brief and all client-stack data are mock. Using real client briefs would need this revisited first.

The rates are a hand-kept table and will go stale; the source URL and the date checked sit next to them.
