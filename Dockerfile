FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

RUN useradd --create-home app
WORKDIR /app
RUN chown app:app /app
USER app

ENV UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# The project installs editable, so the MCP adapter finds mcp_server/ at /app (REPO_ROOT).
COPY --chown=app:app src ./src
COPY --chown=app:app mcp_server ./mcp_server
COPY --chown=app:app ui ./ui
RUN uv sync --frozen --no-dev

CMD ["uvicorn", "creative_ops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
