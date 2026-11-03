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
│   ├── chat_api/                  # today's routers; owns conversations
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
services/chat_api/
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

#### Why the name appears twice

`services/chat_api/chat_api/` looks redundant but is not: the outer directory is the
**project** (a deployable unit — pyproject, Dockerfile, migrations, `.env`) and the inner
one is the **import package**. They are different things that happen to share a name, which
is the standard Python flat layout — `requests/requests/`, `flask/flask/`,
`django/django/`.

Two alternatives were tested and rejected:

| Alternative | Why not |
| --- | --- |
| Project directory *is* the package (`services/chat_api/app.py` importable as `chat_api.app`) | uv's build backend rejects it — *"Module root must be inside the project"*. Achievable only by switching that service to hatchling with a hand-maintained `force-include` manifest, which fails **silently**: add a `stores/` package, forget the manifest, and local dev still works while the container lacks the module |
| Inner package named for the domain (`services/chat_api/chat/`) | Works, but every service must invent and remember a second distinct name, and top-level package names must stay globally unique across the workspace or they collide in a shared environment |

Use `src/` layout (`services/chat_api/src/chat_api/`) instead if you prefer the stricter
convention; it adds nesting rather than removing it, so this starter keeps the flat form.

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
]

[tool.ruff]
lint.ignore = ["E501"]
lint.extend-select = ["I"]
```

Keeping `ruff` configuration at the root means every package is linted identically — one of
the quiet benefits of a workspace.

### A service referencing `base`

```toml
# services/chat_api/pyproject.toml
[project]
name = "chat_api"
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
| Install one service only | `UV_PROJECT_ENVIRONMENT=.venvs/chat_api uv sync --package chat_api` |
| Run a service | `UV_PROJECT_ENVIRONMENT=.venvs/chat_api uv run --package chat_api serve` |
| Add a dep to one service | `uv add --package chat_api some-lib` |
| Add a dep to the library | `uv add --package base --optional llm some-lib` |
| Run one service's migrations | `cd services/chat_api && uv run alembic upgrade head` |
| Lint everything | `uv run ruff check .` |

### A workspace has one `.venv` — use `UV_PROJECT_ENVIRONMENT` for per-service installs

This trips people up, so it is worth stating plainly: **a uv workspace resolves to a single
shared environment.** `uv sync --package chat_api` does not add `chat_api` to what is
already installed — it makes the environment contain *exactly* `chat_api`'s closure, which
**uninstalls** any other service that was there. Two services therefore cannot run locally
out of the same environment.

Point each service at its own environment directory instead:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/chat_api uv sync --package chat_api
UV_PROJECT_ENVIRONMENT=.venvs/identity uv sync --package identity
```

The services now coexist, each with only its own dependency closure, still resolved from
the single root `uv.lock`. Verified end to end: a `base[redis]` consumer imports `redis`
and fails on `qdrant_client`; a `base[vectors]` consumer is the exact inverse; both share
the identical `httpx` build. That is the extras design in [doc 02](02-library-boundary.md)
working as intended.

`.venvs/` is gitignored. The default `.venv` stays useful for whole-workspace work
(`uv sync --all-packages`, linting, repo-wide tooling).

> **Task-runner note for [doc 06](06-build-and-deploy.md):** whatever `make dev s=<svc>`
> ends up being must export `UV_PROJECT_ENVIRONMENT=.venvs/$(s)` so the isolated
> environment is the default path and nobody has to remember this.

`uv sync --package chat_api` installs **only** that service's dependency closure. In a
container that single-closure behaviour is exactly what is wanted — the image has one
service in it — which is the mechanism behind small per-service images in
[doc 06](06-build-and-deploy.md).

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
