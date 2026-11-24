---
paths:
  - "tests/**"
  - "evals/**"
---

# Testing and evals

## Unit tests (`tests/`)
- Never call the real Anthropic API. Use the shared mocked-client fixture in
  `tests/conftest.py`. Unit tests must run offline, fast and the same every time.
- Real API calls are allowed only in `evals/` and in tests marked
  `@pytest.mark.integration`. Those don't run by default.
- Mirror `src/`: `src/agent/parser.py` → `tests/agent/test_parser.py`.
- One behaviour per test. Name it `test_<unit>_<condition>_<expected>`,
  e.g. `test_parse_brief_missing_budget_returns_escalation`.
- Cover the failure paths too: malformed model output, tool errors, timeouts.

## Evals (`evals/`)
- Each case has `id`, `input`, `expected` and `tags`.
- Cover these brief types at minimum: complete, vague, contradictory,
  missing budget, rights conflict. Add a case whenever a real failure shows up.
- Graders, in order of preference:
  1. Code graders: schema validity, field accuracy, correct escalation,
     tool-call correctness (right tool, valid args).
  2. Hallucination check: every output field must trace back to the brief or
     a tool result.
  3. LLM judge, only for what code can't check (tone, reasoning quality).
     Use a written rubric and a pinned model. The API no longer accepts
     `temperature`, so verdicts can vary slightly between runs.
- After any change to a prompt, agent or tool, run the evals and compare
  against the last saved baseline. Report the pass rate per tag and list
  any regressed case IDs.
- Never change expected outputs, loosen graders or delete cases to make an
  eval pass. If you think an expected output is wrong, stop and flag it to
  me with your reasoning.