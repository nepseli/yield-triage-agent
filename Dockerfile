# syntax=docker/dockerfile:1
# Multi-stage image for the yield-triage MCP server and CLI.
# - Base images pinned by tag AND digest.
# - Dependencies installed from uv.lock (--frozen), no dev tools in the runtime.
# - Runs as a non-root user; no secrets are copied or baked in. The approval
#   key, if needed inside a container, must be provided at run time.

FROM ghcr.io/astral-sh/uv:0.12.18@sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc AS uv

FROM python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d AS runtime
RUN groupadd --system app && useradd --system --gid app --no-create-home --home-dir /app app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY scripts ./scripts
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YIELD_TRIAGE_DATA_DIR=/app/data/synthetic \
    YIELD_TRIAGE_STATE_DIR=/app/state
RUN mkdir -p /app/data /app/state && chown app:app /app/data /app/state
USER app
# The server speaks MCP over stdio, so there is no port to probe. The check
# proves the package imports and the shipped policy file validates.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "from yield_triage.policy.config import load_policy; import yield_triage.server.app; load_policy()" || exit 1
CMD ["yield-triage", "serve"]
