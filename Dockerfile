# One image for every service (API, web UI, mock payroll, set-up job); compose picks the
# command. Two stages: build the virtualenv with uv, then copy it into a slim runtime.

# --- build ---------------------------------------------------------------------------------
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is reused until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime -------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 hrm
WORKDIR /app
COPY --from=build --chown=hrm:hrm /app /app
COPY --chown=hrm:hrm data/docs ./data/docs
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HRM_LOG_FORMAT=json \
    HRM_MEMORY_DB_PATH=/app/state/memory.sqlite
# Only these are writable by the app user: model cache, policy index, conversation store.
RUN mkdir -p /app/.cache /app/data/index /app/state \
    && chown hrm:hrm /app/.cache /app/data /app/data/index /app/state
USER hrm

# Bake the policy index and both local models into the image: containers start without
# downloads, and every container of one image answers from the same index.
# `hrm search` loads the reranker once so it is cached too.
RUN hrm ingest && hrm search "warm up" -k 1 > /dev/null

EXPOSE 8000 8501 8090
CMD ["hrm", "serve", "--host", "0.0.0.0"]
