---
status: accepted (revisit in build step 5)
---

# Tool calls in one response run concurrently

When one assistant response contains several `tool_use` blocks, the agent loop runs them all at once with `asyncio.gather` and returns their results in call order. The API treats tool calls in a single response as independent, and the model cannot chain them: a call that needs another call's output (for example, `update_status` needing the ID from `create_task`) has to wait for a later Iteration. That makes concurrency safe for every tool we have today, and it saves wall-clock time on read-heavy steps such as DAM searches during review.

## Consequences

Two writes that touch the same resource in one response, such as two `tracker.update_status` calls on the same task, take effect in no defined order. We are not serialising anything yet, because no write tools exist and a per-tool flag would be unused configuration. When the tracker and DAM write tools and their gates arrive in build step 5, decide whether writes run one at a time in call order. Do not make the whole loop sequential to fix this without revisiting this decision.
