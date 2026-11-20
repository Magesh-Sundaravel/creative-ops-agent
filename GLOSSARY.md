# creative-ops-agent

Takes a fashion/lifestyle creative brief from intake to agency handoff, using agents that act on a mock client stack.

## Language

### Agent execution

**Run**:
One complete execution of an agent, from the first prompt until it finishes, is truncated, is refused, or hits the iteration cap.
_Avoid_: Turn, session, conversation

**Iteration**:
A single model call within a Run, together with any tool calls that call requests.
_Avoid_: Turn, step, round
