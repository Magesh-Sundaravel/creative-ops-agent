---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets.

If the user passes a ticket reference, fetch it from the issue tracker and state its title before starting. If the reference is ambiguous, ask.

Call the Skill tool with "tdd" where possible, at pre-agreed seams.

Run typechecking regularly, single test files regularly, and the full test suite once at the end.

Once done, call the Skill tool with "code-review" to review the work.

Triage every finding: fix it in this ticket, defer it to a named sibling ticket, or skip it with a reason.

- Fix the root cause, not the symptom. If several findings share a cause, fix the cause once.
- Make all the fixes from one review as one batch, not one change per finding.

If you change any code after a review, run "code-review" again before committing:

- Review only the diff since the last review, not the whole branch.
- Tell it which findings are deferred (and to which ticket) or skipped, so it doesn't report them again.
- Stop when a review finds nothing new in this ticket's scope. Commit only once the latest review covers every line you are committing.

Commit your work to the current branch.
