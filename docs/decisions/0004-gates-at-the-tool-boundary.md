# Gates live at the tool boundary

A subagent can write to the client stack only through a gated wrapper around the MCP tool, never through the raw tool and never on the strength of a prompt instruction. Each agent sees only the MCP tools on its allowlist. The Gate checks preconditions (for `tracker_create_task`, an `approved` Review Verdict), allows a fixed number of successful writes per Run (one create), and fills in the arguments code already knows (brand slug, campaign, due date), so the model supplies only what it has to write itself (title and description). A refused write returns a `permission_denied` result that the model can see. The agent's code also refuses to start a Run that the Gate would block, so the Gate is a second layer and not the only check.

## Consequences

The Gate takes its write slot synchronously, before its first `await`. If one response contains two calls to the same write, the second gets `permission_denied` even though the loop runs tool calls concurrently. Writes are therefore serialised per Gate and not by the loop, which settles the open question in ADR 0002.
