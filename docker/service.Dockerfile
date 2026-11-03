# syntax=docker/dockerfile:1.7
# One build for every service. Select with --build-arg SERVICE=<name>.
#   docker build -f docker/service.Dockerfile --build-arg SERVICE=chat_api .

ARG PYTHON_VERSION=3.14

# Every workspace manifest, and nothing else. uv --locked validates the lock
# against the whole workspace, so a build that saw only one service's
# pyproject.toml would fail as soon as a second service existed.
FROM busybox:1.37 AS manifests
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY services/ services/
RUN find packages services -type f ! -name pyproject.toml -delete && \
    find packages services -type d -empty -delete


FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG SERVICE

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=manifests /src/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-workspace --no-editable --package ${SERVICE}

# Only this service's source.
COPY packages/ packages/
COPY services/${SERVICE}/ services/${SERVICE}/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package ${SERVICE}

# Drop other services' manifests now the lock has been validated, so the
# runtime image carries nothing belonging to another service.
RUN find services -mindepth 1 -maxdepth 1 -type d ! -name "${SERVICE}" -exec rm -rf {} +


FROM python:${PYTHON_VERSION}-slim-bookworm AS runner
ARG SERVICE

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /app /app
COPY --chown=appuser:appgroup docker/run.sh /app/docker/run.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SERVICE=${SERVICE} \
    API_HOST=0.0.0.0 \
    API_PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD ["python", "-m", "base.health_check"]

ENTRYPOINT ["/bin/sh", "/app/docker/run.sh"]
