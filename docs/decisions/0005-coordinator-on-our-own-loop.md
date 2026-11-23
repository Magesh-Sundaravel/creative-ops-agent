# The Coordinator runs on our own loop, not the Claude Agent SDK

The Coordinator is an LLM agent on the same hand-written loop as the subagents (ADR 0001). Its tools are the Intake, Review and Handoff agents. It decides which to call next, and code decides when to escalate. We are not using the Claude Agent SDK for orchestration, even though the original stack listed it.

## Considered Options

- **Code pipeline.** Simple and deterministic, but "decomposition" and "re-plan" would only be `if` statements, so the step would teach nothing about coordination.
- **Claude Agent SDK.** It would run the Coordinator in a different runtime from the subagents. We would lose the single message history per Run that traces and evals depend on, and the iteration cap and `is_error` handling we already test.

## Consequences

Parallel subagents come from the loop's concurrent tool calls (ADR 0002), not from an SDK feature. Escalations are triggered in code from typed agent results, never by the model.
