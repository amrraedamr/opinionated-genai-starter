# 01 — Workspace Layout

## Decision

Use a **uv workspace monorepo**: one `uv.lock` at the root, many independently buildable
packages underneath.

uv 0.12.19 is already in use with a committed `uv.lock`, so this is a natural extension
rather than a new tool.

### Why a workspace rather than separate repos

| Concern | Workspace | Separate repos |
| --- | --- | --- |
| Changing `base` + a consumer together | One commit, one PR | Publish, bump, two PRs, version skew |
| Transitive version consistency | Guaranteed — single resolution | Drifts silently per repo |
| Cross-service refactor | Mechanical, verifiable | Coordinated migration |
| Onboarding a new service | Copy a template directory | Repo setup, CI setup, access |
| Independent deploy cadence | Preserved (per-service images) | Preserved |

The common objection to monorepos — *"you lose independent deployability"* — does not apply.
Deployability is determined by what each container image contains and what each service
owns at runtime, not by source layout. [Doc 06](06-build-and-deploy.md) keeps images
per-service.

### Why not separate uv projects without a workspace

Each project would get its own lockfile, so `httpx` could resolve to different patch
versions in different services while they share `base`. That is the exact class of bug a
shared foundation is supposed to eliminate.

---

## Target directory structure

```
opinionated-genai-starter/
├── pyproject.toml                 # workspace root — members only, no runtime deps
├── uv.lock                        # ONE lockfile for every package
├── .env.shared.example            # local-dev-only shared infra (see doc 04)
├── docker/
│   ├── service.Dockerfile         # one parameterised build for all services
│   └── base.dockerignore
├── deploy/
│   ├── docker-compose.yml         # full local stack
│   └── docker-compose.infra.yml   # just Postgres/Redis/Mongo/Qdrant
├── docs/architecture/             # these documents
│
├── packages/                      # shared, non-deployable libraries
│   ├── base/                      # the foundation (today's base/, domain-free)
│   │   ├── pyproject.toml         # core deps + optional extras
│   │   └── base/
│   │       ├── app.py             # create_app() factory
│   │       ├── clients/
│   │       ├── models/            # definitions + view models + ORM *mixins* only
│   │       ├── services/
│   │       └── tools/
│   └── contracts/                 # cross-service DTOs + event schemas
│       ├── pyproject.toml         # depends only on pydantic
│       └── contracts/
│           ├── events/
│           └── v1/
│
├── services/                      # deployable units — each becomes an image
│   ├── identity/                  # owns users, login, OAuth, token issuing
│   ├── chat-api/                  # today's routers; owns conversations
│   └── ingestion-worker/          # no HTTP server; broker consumer
│
└── tools/
    ├── new_service.py             # scaffolds a service from a template
    └── affected.py                # which services changed (for CI)
```

### Anatomy of one service

Every service is the same shape. This uniformity is what makes scaffolding and a single
Dockerfile possible.

```
services/chat-api/
├── pyproject.toml          # name, its own dependency subset, base extras
├── .env.example            # documented; committed
├── .env                    # real values; gitignored (already covered by .gitignore)
├── Dockerfile              # 3 lines — delegates to docker/service.Dockerfile
├── alembic.ini             # only if this service uses SQL
├── alembic/
│   └── versions/           # this service's migrations ONLY
└── chat_api/
    ├── __init__.py
    ├── app.py              # create_app(...) — composition root
    ├── settings.py         # Settings composed from base mixins
    ├── routers/            # auto-discovered by init_routers
    ├── models/
    │   ├── orm/            # this service's tables, own DeclarativeBase
    │   └── view_models/    # this service's request/response models
    ├── services/           # this service's business logic
    └── stores/             # AbstractStore implementations over its own DB
```

> A service that uses MongoDB has no `alembic*` and no `orm/` — it has `documents/`
> instead. A worker service has no `routers/`. The template omits what is not needed.

---

## Workspace wiring

### Root `pyproject.toml`

The root is a coordination file, not a package. It declares members and shared tooling
configuration only.

```toml
[tool.uv.workspace]
members = ["packages/*", "services/*"]

[tool.uv]
default-groups = "all"

[dependency-groups]
dev = [
    "pre-commit>=2.16.0",
    "pre-commit-hooks>=4.1.0",
    "ruff>=0.15.4,<0.16",
    "pytest>=8",
    "pytest-asyncio>=0.24",
]

[tool.ruff]
lint.ignore = ["E501"]
lint.extend-select = ["I"]
```

Keeping `ruff` configuration at the root means every package is linted identically — one of
the quiet benefits of a workspace.

### A service referencing `base`

```toml
# services/chat-api/pyproject.toml
[project]
name = "chat-api"
version = "0.1.0"
requires-python = ">=3.12,<3.15"
dependencies = [
    "base[sql,postgres,llm,redis,auth,server]",
    "contracts",
]

[project.scripts]
serve = "chat_api.app:start"

[tool.uv.sources]
base = { workspace = true }
contracts = { workspace = true }

[tool.uv.build-backend]
module-root = ""

[build-system]
requires = ["uv_build>=0.12.19,<0.13.0"]
build-backend = "uv_build"
```

`[tool.uv.sources]` with `workspace = true` is what makes uv resolve `base` from the local
checkout (editable during development) instead of from PyPI.

---

## Everyday commands

| Task | Command |
| --- | --- |
| Install everything | `uv sync --all-packages` |
| Install one service only | `UV_PROJECT_ENVIRONMENT=.venvs/chat-api uv sync --package chat-api` |
| Run a service | `UV_PROJECT_ENVIRONMENT=.venvs/chat-api uv run --package chat-api serve` |
| Add a dep to one service | `uv add --package chat-api some-lib` |
| Add a dep to the library | `uv add --package base --optional llm some-lib` |
| Run one service's migrations | `cd services/chat-api && uv run alembic upgrade head` |
| Lint everything | `uv run ruff check .` |

`uv sync --package chat-api` installs **only** that service's dependency closure. This is
the mechanism behind small per-service images in [doc 06](06-build-and-deploy.md).

> **A workspace has one shared `.venv`.** Plain `uv sync --package svc-b` *uninstalls*
> `svc-a`, so two services cannot run locally from the same environment. Set
> `UV_PROJECT_ENVIRONMENT=.venvs/<service>` to get coexisting per-service environments, and
> add `.venvs/` to `.gitignore`.
>
> Verified: with `base` exposing `redis` and `vectors` extras, a service declaring
> `base[redis]` imports `redis` and raises `ModuleNotFoundError` on `qdrant_client`, while a
> service declaring `base[vectors]` is the exact inverse — both sharing the core library's
> `httpx`, under a single `uv.lock`.

---

## Degrading to a single application

A user who wants today's experience runs the scaffolder once and gets a workspace with one
member. The structure is identical; there is simply one directory under `services/`. No
separate "monolith mode" is maintained, which avoids two divergent starter paths.

---

## Open questions

- **Should `contracts` be one package or one per service pair?** Start with one; split only
  if it becomes a change-coupling bottleneck.
- **Should `base` be publishable to a private index?** Not initially. Document it as the
  escape hatch for a service that must pin an older `base` — see
  [doc 07](07-modularity-backlog.md).
