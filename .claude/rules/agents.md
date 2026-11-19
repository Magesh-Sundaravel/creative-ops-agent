---
paths:
  - "src/creative_ops/agents/**"
  - "src/creative_ops/loop/**"
  - "src/creative_ops/tools/**"
  - "src/creative_ops/guards/**"
  - "src/creative_ops/schemas/**"
---

# Agent design rules

## Loop
- Continue on `stop_reason == "tool_use"`, finish on `"end_turn"`. Never inspect text to decide.
- Hitting `max_iterations` raises an error. It's a safety net, not a normal exit.
- Every `tool_use` gets a `tool_result` with the matching `tool_use_id`. Failures too,
  returned as a structured error with `is_error: true`, never swallowed.

## Multi-agent
- Subagents don't inherit coordinator history. Pass each one only the context it needs.
- Each subagent gets only the tools its job needs.
- Run independent subagents in parallel (`asyncio.gather`).
- If review coverage has gaps, the coordinator re-plans them. Never report success with gaps open.

## Structured output
- Extract via tool use with a JSON schema, then validate with Pydantic.
- Fields the brief may not contain are `Optional` and stay `None`. Never invent values.
  Missing critical fields go in `missing_fields: list[str]`.
- On validation failure, retry once with the error in context, then escalate.

## Gates and escalation
- Costly mistakes (budget, usage rights, sending to production) are blocked in code
  (guards, schemas, hooks), not by prompt instructions.
- Escalate only on explicit criteria: missing critical fields, rights conflict, budget over
  threshold, repeated tool failure. Never on tone or the model's self-reported confidence.
- Recover locally first (retry retryable errors, re-query), then escalate with full context.

## Loop vs SDK
- `loop/agent_loop.py` stays framework-free (raw `anthropic` SDK).
- Use the Agent SDK only for subagent orchestration and hooks. Record each use as an ADR
  in `docs/decisions/`.