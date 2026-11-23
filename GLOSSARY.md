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
The fields of one campaign in a Brief, holding only what the Brief states and nothing inferred. A Brief has one or more.
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

### Client stack

**Asset**:
A finished or in-progress piece of creative work held in the DAM, described by its metadata only.
_Avoid_: File, creative, artwork

**Usage Rights**:
The territories, channels and time period in which an Asset may be published.
_Avoid_: License, permissions, clearance

**Brand Guidelines**:
A brand's rules for how its creative work must look and sound, such as palette, typography, logo use, tone and mandatory legal lines.
_Avoid_: Brand book, style guide, brand rules

**Task**:
A unit of agency work tracked in the project tracker, moving through a fixed set of statuses until it is delivered.
_Avoid_: Ticket, job, card

### Review and handoff

**Finding**:
One problem Review reports, either with an Asset or with a deliverable that has no matching Asset. It is either blocking or advisory.
_Avoid_: Issue, flag, violation

**Review Verdict**:
The result of a review: approved when there are no blocking Findings, and revisions requested otherwise.
_Avoid_: Decision, outcome, status

**Handoff Package**:
Everything production needs to deliver an approved campaign: which Assets cover each deliverable, the Usage Rights, mandatory legal lines, the deadline, notes, and the Task created for the work.
_Avoid_: Handover, delivery pack, output

**Gate**:
A check that a write to the client stack has to pass before it happens, regardless of what the agent intends.
_Avoid_: Guard, permission, approval

### Coordination

**Coordinator**:
The agent that takes a Brief through intake, review and handoff, deciding which agent runs next from what each one returns.
_Avoid_: Orchestrator, supervisor, manager

**Escalation**:
The Coordinator stopping and handing a decision to a human, with the reason, what it needs, and every Ambiguity in the Brief.
_Avoid_: Escalation gate, human gate, approval
