---
paths:
  - "mcp_server/**"
  - "src/creative_ops/tools/**"
---

# Tools and MCP

## Tool descriptions
The model picks tools from their descriptions alone, so every description must state:
- what the tool does (one sentence)
- when to use it, and when NOT to (name the tool to use instead)
- each input: meaning, format, valid values or range

## Least privilege
- Give each subagent only the tools its job needs. Intake is read-only: no write tools.
- When you add a tool, also update which subagents can use it. Never give it to every subagent by default.

## Errors
- Tools never raise a bare exception or return an error string. They return
  `ToolError{type, message, retryable, details}`.
- `type` is one of: `not_found`, `invalid_input`, `permission_denied`, `upstream_unavailable`.
- `retryable` is true only for `upstream_unavailable`. The agent uses it to decide whether to retry.
- `message` is written for the model: say what went wrong and what to do next.

## Resources vs tools
- Read-only reference data (brand guidelines, templates) is exposed as an MCP resource, not a tool.
- Tools are for actions or for data that changes per call.

## Credentials
- MCP credentials are read from env vars only.
- Never put them in prompts, tool output, error `details`, or logs.

## Seed data
- Seed data is fixed test input. Ask before changing it.