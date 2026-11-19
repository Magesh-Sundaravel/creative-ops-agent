---
paths:
  - "**/*.py"
---

# Python conventions

- async/await for all I/O (Claude API, MCP, HTTP).
- Type hints on every function. Docstrings on every function outside `tests/`.
- Data crossing a module or LLM boundary is a Pydantic model, never a raw dict.
  Generate tool `input_schema` from the model (`model_json_schema()`), don't hand-write it.
- Secrets and model IDs come from `config.py` (env vars). Never hardcode or inline them.
  Never log or print secrets.
- Log structured JSON. Include `run_id` and `agent`, plus `step`, `tool`, `latency_ms`
  and token usage where they apply.
- Raise specific exceptions. Catch broad exceptions only at boundaries (API routes,
  the agent loop's tool dispatch), and always log or convert them. Never `except Exception: pass`.