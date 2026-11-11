# 06 — Build & Deploy

One build definition, N images. Per-service deployability without per-service build drift.

---

## Current build

`docker/service.Dockerfile` is parameterised by service. It validates the
workspace lock, installs only the selected service's dependency closure, and
removes the other service sources before producing the runtime image.

It syncs with `--no-dev`, so a service's `dev` dependency group stays out of the
image. The three SQL services here run PostgreSQL in production and reach for SQLite
only as a local default, so that is where their SQLite driver lives. A service
that genuinely runs SQLite in production declares `base[sql,sqlite]` and ships it.

---

## One parameterised Dockerfile

`uv sync --package <name>` inside a workspace installs only that package's dependency
closure. That is the whole mechanism.

```dockerfile
# docker/service.Dockerfile
# syntax=docker/dockerfile:1.7
ARG SERVICE

# ---------- builder ----------
FROM python:3.14-slim-bookworm AS builder
ARG SERVICE

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer: manifests + lock only, so it caches across source edits.
COPY pyproject.toml uv.lock ./
COPY packages/base/pyproject.toml      packages/base/
COPY packages/contracts/pyproject.toml packages/contracts/
COPY services/${SERVICE}/pyproject.toml services/${SERVICE}/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --package ${SERVICE}

# Source layer: only what this service can legitimately import.
COPY packages/ packages/
COPY services/${SERVICE}/ services/${SERVICE}/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package ${SERVICE}

# ---------- runner ----------
FROM python:3.14-slim-bookworm AS runner
ARG SERVICE

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app/services/${SERVICE}

COPY --from=builder --chown=appuser:appgroup /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    SERVICE_NAME=${SERVICE} \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "base.health_check"]

ENTRYPOINT ["/bin/sh", "/app/docker/run.sh"]
```

Notes on the deltas from today's Dockerfile:

- **`COPY services/${SERVICE}/` only.** A service's image physically cannot contain another
  service's source, which turns the data-ownership rule into a build-time guarantee.
- **BuildKit cache mounts** make the second and subsequent service builds fast, since they
  share `base`'s wheels.
- **`WORKDIR` is the service directory**, so relative paths in `env_file` and `alembic.ini`
  resolve exactly as they do in local development.
- **`start-period`** added — the current healthcheck can fail a slow first boot.
- `health_check.py` moves into `base` so every service inherits it.

Each service keeps a three-line `Dockerfile` for tools that expect one in-place:

```dockerfile
# services/chat_api/Dockerfile
# Build from the repo root: docker build -f docker/service.Dockerfile --build-arg SERVICE=chat_api .
```

### Entrypoint

Today's `run.sh` generalises with no loss of behaviour:

```sh
#!/bin/sh
set -e

python -m base.preflight

if [ -n "$RUN_MIGRATIONS" ] && [ -f alembic.ini ]; then
    echo "Running migrations for ${SERVICE_NAME}..."
    alembic upgrade head
fi

export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

echo "Starting ${SERVICE_NAME}"
exec serve
```

`serve` is each service's own `[project.scripts]` entry ([doc 01](01-workspace-layout.md)),
so one entrypoint serves API services and workers alike. The `-f alembic.ini` guard means
NoSQL and stateless services need no special case.

### Startup preflight

A subsystem's capability is fixed at build time by the extras its service declares.
Which implementation it uses is chosen later, by environment variables. Nothing
reconciles the two, so configuration can name something the image cannot honour.

`python -m base.preflight` closes that. `run.sh` runs it before migrations and before
`serve`, for every service:

```
Preflight failed: SQL_BACKEND='sqlite' needs aiosqlite, which this build does not
install. This build supports: postgresql. Either set SQL_BACKEND to one of those, or
deploy an image built with "base[sqlite]" (installs aiosqlite).
```

It names both remedies, because both audiences hit this: a developer adds the extra, an
operator reverts the configuration. Running it before `serve` also makes the failure
deterministic — `SQLClient._require_driver` already raised, but *when* depended on whether
a service built its stores eagerly at import or lazily on first use, so one
misconfiguration could be a boot crash in one service and a 500 on first request in
another.

Two checks today:

| Check | Scope | Unknown value |
| --- | --- | --- |
| `SQL_BACKEND` | builds carrying the `sql` extra | fatal |
| `BROKER` | every service | fatal |

The `sql` extra, not `alembic.ini`, is what scopes the SQL check. `alembic.ini` marks
migrations, so gating on it would skip a service that owns SQL tables whose schema is
managed outside the repository — and the two are independent, since a MySQL service may
well carry migrations too. A build with no `sql` extra cannot speak SQL at all, so
`SQL_BACKEND` is inert there and the check is skipped.

**An unknown `BROKER` is fatal on purpose.** `base.services.broker.factory.build_broker`
dispatches exhaustively, but before it was centralised each service carried its own copy
that treated anything other than an exact `redis` match as in-process. `BROKER=Redis` — or
a trailing space out of a secret store — started cleanly, passed its readiness probe
(`InMemoryBroker` is "reachable whenever the object exists") and then delivered nothing
between containers. A silent, green-health-check message loss is worse than a crash, so
the value is validated rather than defaulted, in three places: `BrokerBackend` is a
`StrEnum` so settings loading rejects it, the preflight rejects it before `serve`, and the
factory raises on a transport it has no case for.

`BROKER=memory` outside `ENVIRONMENT=local` warns rather than fails, because a deliberate
single-process deployment is legitimate.

Ask an image what it supports without starting it:

```
docker run --rm --entrypoint python registry/chat_api:<sha> -m base.preflight
```

CI asserts both directions per service: the image accepts the configuration it was built
for, and rejects configuration it cannot honour.

> **Migrations in production:** `RUN_MIGRATIONS` on container start is convenient but races
> when a service scales to N replicas. Prefer a pre-deploy job or init container that runs
> `alembic upgrade head` exactly once; keep the env flag for local and single-replica use.

### `.dockerignore`

```
.git
.venv
**/.venv
**/__pycache__
**/.env
**/.env.*
!**/.env.example
**/keys
**/*.db
docs
```

Excluding `**/.env` and `**/keys` is a security control, not tidiness — assert it in CI
by grepping a built image for `.env` files.

---

## Local orchestration

Split infrastructure from services so day-to-day work does not rebuild containers.

```yaml
# deploy/docker-compose.yml
name: genai-starter

services:
  identity:
    build:
      context: ..
      dockerfile: docker/service.Dockerfile
      args: { SERVICE: identity }
    env_file: [../.env.shared, ../services/identity/.env]
    environment:
      RUN_MIGRATIONS: "1"
    ports: ["8001:8000"]
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_healthy }

  chat_api:
    build:
      context: ..
      dockerfile: docker/service.Dockerfile
      args: { SERVICE: chat_api }
    env_file: [../.env.shared, ../services/chat_api/.env]
    environment:
      RUN_MIGRATIONS: "1"
      IDENTITY_BASE_URL: http://identity:8000
    ports: ["8002:8000"]
    depends_on:
      identity: { condition: service_started }
      postgres: { condition: service_healthy }
```

`deploy/docker-compose.infra.yml` holds Postgres (with the `init-dbs.sql` seed from
[doc 03](03-data-ownership.md)), Redis, Mongo and Qdrant, each with a real healthcheck so
`depends_on: service_healthy` works.

A developer working on one service runs infra in Docker and that single service on the
host via `uv run --package chat_api serve` — fast reload, real dependencies.

---

## CI: build only what changed

With N services, rebuilding everything on every commit wastes most of the budget. But
`base` is a dependency of all of them, so a change there legitimately rebuilds everything.

```
changed files
   ├── packages/base/**      → every service
   ├── packages/contracts/** → every service that depends on it
   ├── services/<x>/**       → only service x
   ├── uv.lock               → every service
   └── docs/**               → nothing
```

`tools/affected.py` resolves this from `git diff --name-only` plus each service's declared
workspace dependencies, and emits a JSON list for a GitHub Actions matrix:

```yaml
jobs:
  detect:
    outputs:
      services: ${{ steps.affected.outputs.services }}
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - id: affected
        run: echo "services=$(uv run python tools/affected.py)" >> "$GITHUB_OUTPUT"

  build:
    needs: detect
    if: needs.detect.outputs.services != '[]'
    strategy:
      matrix:
        service: ${{ fromJson(needs.detect.outputs.services) }}
    steps:
      - run: uv sync --package ${{ matrix.service }}
      - run: uv run --package ${{ matrix.service }} ruff check .
      - run: docker build -f docker/service.Dockerfile --build-arg SERVICE=${{ matrix.service }} .
```

Repository-wide gates that run regardless:

| Check | Why |
| --- | --- |
| `uv lock --check` | Lockfile matches manifests |
| `ruff check .` | One config at the root, uniform across packages |
| **Boundary lint** | No service imports another service; `base` imports no service |
| **Dependency isolation lint** | No service image carries another service's dependencies |
| Image secret scan | No `.env` or key material in any built image |

The boundary lint is the one worth writing by hand — it is the mechanical enforcement of
[doc 02](02-library-boundary.md) and [doc 03](03-data-ownership.md), and without it the
architecture erodes one convenient import at a time.

`tools/deps_lint.py` does the same job for the dependency graph. `tools/import_lint.py`
covers what neither can: both of those reason about declarations, while it runs the
import statements with every extra installed. A module behind an extra that no shipped
service declares is never loaded by the per-service jobs, so a stale import inside it
would otherwise reach main unnoticed. `uv sync --package` keeps
images separate only while each service declares the extras it actually uses, so the lint
checks three things: a service declares no extra `base` does not define; a service's
`--no-dev` closure contains nothing belonging solely to an extra it did not declare; and
`base`'s core dependency set still matches a pinned list. The first two derive their deny
list from `base`'s own manifest, so a new extra is covered the day it is added. The third
is pinned on purpose — a core dependency lands in every image ever built, so growing that
set should be a decision someone reviews rather than a side effect of one import.

"Solely" is judged from the resolved graph in `uv.lock`, not from the extras' own lists.
The `mcp` SDK requires `sse-starlette`, which the `sse` extra also lists; a service
declaring `mcp` ships it for `mcp`'s sake, and that is not a leak of `sse`. A package that
only an undeclared extra needs is still refused.

The image job preflights each image with configuration that image could honour. It asks
the image which broker it installs and whether it speaks SQL, rather than assuming
`BROKER=redis` and a SQL backend everywhere, so a service with neither — `mcp_server` —
is checked against what it can actually be given.

---

## Deployment shape

Each service is an independent unit: its own image, its own scaling, its own configuration,
its own database.

| Concern | Per service |
| --- | --- |
| Image | `registry/<service>:<sha>` |
| Scaling | Independent — a worker scales on queue depth, an API on RPS |
| Config | Own environment variables and secret references ([doc 04](04-configuration.md)) |
| Database | Own instance or own logical database ([doc 03](03-data-ownership.md)) |
| Migration | Own pre-deploy job |
| Rollback | Independent, because schemas are expand/contract compatible |

Tag images with the commit SHA so one commit maps to a reproducible set — the deploy-time
benefit of the single lockfile.

### Liveness and readiness

Today there is one static `/health-check` returning `{"status": "ok"}`, which reports
healthy while the database is unreachable.

| Probe | Answers | Behaviour |
| --- | --- | --- |
| `/livez` | Is the process up? | Static. Failure → restart |
| `/readyz` | Can it serve traffic? | Queries each enabled capability's `probe()`. Failure → remove from load balancer |

The capability registry from [doc 02](02-library-boundary.md) supplies `/readyz` for free:
each enabled subsystem reports whether its backing dependency is reachable, and a service
with no database simply has no database probe.

### Graceful shutdown

This starter streams SSE (`EventSourceResponse`, `StreamIterator`) and scopes instances by
request via `Stored`. On rolling deploys the service must stop accepting connections, let
in-flight streams finish within `graceful_timeout` (gunicorn is already configured with
300s), and only then exit — otherwise every deploy severs active chat streams.

Verify per service that `Stored.delete_scope()` still runs for streams terminated by
shutdown rather than by response completion.

## Production topology

Build and deploy `chat_api`, `identity`, `ingestion_worker` and `mcp_server` independently. Run one migration Job per service database before rollout, never on every replica — concurrent migrations race. Supply managed PostgreSQL and Redis dependencies, platform secrets, readiness/liveness probes, and graceful termination. `chat_api` gets only the public verification key; the private signing key is scoped to `identity`.

This repository deliberately ships no platform manifests. Kubernetes, ECS, Container Apps and Nomad disagree about secrets, ingress and autoscaling, so a committed example would be placeholder YAML that rots rather than something you can run. The requirements above are what your platform's manifests have to satisfy.

Compose is local-only. The shared middleware provides single-process backpressure, trusted hosts, request-size limits, security headers and consistent overload responses. Rate budgets count per replica with `RATE_LIMIT_BACKEND=memory` and across the fleet with `redis`; see [09](09-capabilities.md#rate-limiting). Request, auth and connect budgets are keyed by client address, so behind a proxy put an ingress or gateway in front that limits per real client.
