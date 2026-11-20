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

### Briefs

**Brief**:
The client's request for creative work as free text, exactly as received.
_Avoid_: Raw brief, request, job

**Structured Brief**:
The fields of a Brief, holding only what the Brief states and nothing inferred.
_Avoid_: Parsed brief, brief object

**Missing Field**:
A Structured Brief field that the Brief gives no value for.
_Avoid_: Gap, empty field

**Critical Field**:
A Structured Brief field that the agency cannot start work without, so if it is missing a human has to step in.
_Avoid_: Required field, mandatory field

**Ambiguity**:
Something the Brief does say but that is unclear or contradicts itself, such as two different deadlines.
_Avoid_: Note, concern, warning
