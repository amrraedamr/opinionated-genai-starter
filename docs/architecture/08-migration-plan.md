# 08 — Migration Plan

Incremental path from today's single application to the multi-service structure. **The
application runs and is deployable at the end of every phase.** No big-bang rewrite, no
long-lived branch.

---

## Sequencing principle

Order the work so that each phase is independently valuable and independently revertible:

1. **Structure before semantics** — move files while behaviour is identical, so the diff is
   reviewable as "moves only".
2. **Make it possible before making it plural** — extract the factory *before* adding a
   second service, so service two is the first real test of the abstraction.
3. **Split the domain last** — the identity/chat split is the only genuinely risky step,
   and it should happen when everything around it is already stable.

---

## Phase 0 — Workspace scaffold

**Goal:** same code, workspace layout. Pure moves.

- Create `packages/base/`, move `base/` into it with its own `pyproject.toml`.
- Root `pyproject.toml` becomes workspace-only (`[tool.uv.workspace]`, shared `ruff`, dev
  group). Add `src = [".", "packages/base"]` under `[tool.ruff]` so isort still resolves
  `base` as first-party from files outside the package.
- Regenerate `uv.lock`; verify `uv sync --all-packages`.
- Leave the single app's operational files (`alembic/`, `alembic.ini`, `.env.example`,
  `Dockerfile`, `run.sh`) at the repository root. They are still coherent there — the root
  is the only app — and phase 2 moves them once, directly to their final home in
  `services/chat_api/`, rather than relocating them twice.

**Risk:** low. **Verify:** app boots, `/health-check` responds, `alembic upgrade head`
works, existing endpoints behave.

**Reviewer note:** this diff should be ~100% renames. Anything else belongs in a later
phase.

---

## Phase 1 — App factory and webserver parameterisation

**Goal:** `base` can build *an* app, not *the* app.

- Extract `create_app(...)` into `packages/base/base/app.py` from today's `main.py`.
- Add the `Capability` protocol; convert `PromptLoader` and `TerminalCaptureService` into
  capabilities instead of hardcoded lifespan steps.
- `run_webserver(app_path=..., settings=...)` — remove the `"base.main:app"` literal from
  both `run_uvicorn` and `run_gunicorn`.
- Add `/livez` and `/readyz` backed by capability probes.
- Keep `base/main.py` as a thin shim that calls `create_app()` so nothing external breaks.

**Risk:** low–medium — touches startup. **Verify:** lifespan still loads prompts; SSE
`/logs` still streams; gunicorn path still serves.

---

## Phase 2 — First service extraction

**Goal:** the monolith becomes *service number one*. This is the phase that proves the
design.

- Create `services/chat_api/` with `pyproject.toml` depending on `base[...]`.
- Move `base/routers/{auth,conversation,user}.py` → `services/chat_api/chat_api/routers/`.
- Move `base/models/orm/*` and the concrete stores (`store/user.py`, `store/conversation.py`,
  `store/user_identity.py`, `store/chat_history/sql.py`) into the service.
- Move `base/services/auth/` **wholesale** into the service. It imports `UserDTO` and
  `UserStore` directly, so it cannot stay behind once those move; phase 5 extracts the
  stateless verification half back into `base`.
- Move the domain view models (`auth`, `user`, `conversation`, `chat`), the
  `DatabaseTable` enum, the concrete `TestAgent`, and the prompt templates.
- Make `PromptLoader` and `ChatHistoryService` take the service's templates and store
  instead of hardcoding `base`'s — see [doc 02](02-library-boundary.md).
- **Remove `_import_all_tables()`** and its call in `base/clients/sql/client.py`; the
  service's `alembic/env.py` imports its models explicitly.
- Introduce the service's own `DeclarativeBase`; `base` keeps only `TimestampMixin` and
  `SerializableMixin`. Retype `base/tools/serializers.py` against `DeclarativeBase`.
- Move `alembic/`, `alembic.ini` and `.env.example` into the service. Existing migration
  files move unchanged — the schema is identical, so no new migration is generated.
- `base/main.py` shim is deleted here, along with `base`'s console script. The entrypoint
  becomes the service's `serve`.

**Risk:** medium — the metadata change is the subtle part. **Verify:** `alembic
upgrade head` on a *fresh* database produces the identical schema; `alembic check` reports
no pending autogenerate diff against the existing one.

---

## Phase 3 — Dependency extras and settings composition

**Goal:** services declare what they use.

- Split `base` dependencies into extras; add `MissingExtraError` guards at each optional
  import site. `base/tools/serializers.py` degrades rather than raising — it is core.
  `SQLClient` probes its backend's drivers, which SQLAlchemy loads by URL rather than by
  import.
- Split `Settings` into `CoreSettings` + mixins; `chat_api` composes its own. Field names
  are lifted unchanged; only `app_name` → `service_name` is renamed.
- Add `configure()` / `get_settings()`; migrate every `settings` import site in `base` and
  delete the old singleton outright. The composition root and `alembic/env.py` both call
  `configure()` first.
- Add `Protocol` config contracts for SQL, Postgres, MySQL, LLM, embeddings, cache,
  vectors and Azure Search.
- Service `.env.example` + root `.env.shared.example`; document precedence.
- `run.sh` runs from the service directory so its `.env` and `../../.env.shared` resolve.

**Risk:** medium — many call sites, but mechanical and type-checkable. **Verify:**
`uv sync --package chat_api` installs no Qdrant/Cosmos/Mongo/Azure Search; a second service
installed with different extras gets the inverse closure from the same lockfile; the
service boots with only its own `.env`, with precedence shared < `.env` < `.env.local` <
process env proven by test; a deliberately missing extra produces the friendly error.

> After this phase `base` is a genuine library: domain-free, dependency-light, consumable.

---

## Phase 4 — Build, deploy, scaffolding

**Goal:** make adding service two cheap *before* adding service two.

- `docker/service.Dockerfile` parameterised by `ARG SERVICE`; generalised `run.sh`.
- `deploy/docker-compose.yml` + `docker-compose.infra.yml` with per-database seeding.
- `tools/affected.py` + CI matrix; boundary lint; `uv lock --check`.
- `tools/new_service.py` (copier template).

**Risk:** low — additive. **Verify:** `docker compose up` brings up the stack healthy;
scaffolder output builds and boots unmodified.

---

## Phase 5 — Identity split

**Goal:** prove multi-service for real. The first phase with genuine behavioural risk.

- `base/services/auth` splits: stateless `TokenVerifier` + `Principal` stay in `base`;
  login, OAuth, refresh and user storage move to `services/identity/`.
- `user` / `user_identity` tables and stores move to `identity`, with its own database.
- `chat_api` drops those tables and depends on verified claims only; `conversation.user_id`
  becomes an unconstrained `UUID`.
- Replace `validate_user_dependency` with `principal_dependency` at every call site.
- Add `IdentityClient` for the cases needing profile data beyond claims.

**Risk:** highest — it is a data migration plus an auth change. Sequence it as:

1. Deploy `identity` reading the existing shared database (same tables, new service).
2. Switch `chat_api` to claims-based `Principal` while the tables still exist.
3. Only then move the tables to a separate database and drop them from `chat_api`.
4. Drop the cross-table FK **before** separating databases, not after.

**Verify:** tokens issued by `identity` validate in `chat_api`; no `chat_api` code path
queries a user table; both services start with their own databases from empty.

---

## Phase 6 — Async backbone

**Goal:** services that are not request/response.

- `AbstractBroker` + `InMemoryBroker` + `RedisStreamsBroker`.
- `packages/contracts/` with versioned DTOs and event schemas.
- First worker service (`ingestion-worker`) — no HTTP, demonstrating the non-API shape and
  a non-SQL store.
- Correlation ID propagation across HTTP and broker; optional `base[otel]`.

**Risk:** medium, but fully additive — nothing existing changes behaviour.

---

## Summary

| Phase | Theme | Risk | Status |
| --- | --- | --- | --- |
| 0 | Workspace scaffold | Low | Done |
| 1 | App factory | Low–Med | Done |
| 2 | First service extraction | Medium | Done |
| 3 | Extras + settings | Medium | Done |
| 4 | Build, deploy, scaffolding | Low | Done |
| 5 | Identity split | **High** | Done |
| 6 | Async backbone | Medium | Done |

All seven phases are implemented. The workspace holds `base` and
`contracts` under `packages/`, and `chat_api`, `identity` and
`ingestion_worker` under `services/`.

**Phases 0–3 deliver most of the value.** At that point `base` is a real shared library and
adding a second service is a scaffolder invocation. Phases 4–6 are about making the second,
third and fourth service cheap — worth doing when a concrete second service exists, not
speculatively.

### Stopping points

Both of these are legitimate places to stop, not failures:

- **After phase 3** — a properly layered single application with a reusable foundation. If
  microservices never materialise, nothing has been wasted.
- **After phase 5** — true multi-service with independent data ownership. Phase 6 is only
  needed once something genuinely asynchronous appears.

---

## What to decide before starting

| Question | Why it matters |
| --- | --- |
| Which services actually exist in the target? | Phase 5's boundary should follow real ownership, not a guess |
| Is `identity` a separate service, or does one app keep users? | Determines whether phase 5 is needed at all |
| One database engine or several? | Affects extras, compose, and migration tooling |
| Does anything need async messaging today? | If not, defer phase 6 entirely |

---

## Running it

```bash
cp .env.shared.example .env.shared
docker compose -f deploy/docker-compose.yml up --build
```

| Target | Command |
| --- | --- |
| Architectural boundaries | `uv run python tools/boundary_lint.py` |
| Services a change affects | `uv run python tools/affected.py --format table` |
| A new service | `uv run python tools/new_service.py --name billing --storage postgres --http` |
| Port a starter commit | `uv run python tools/port_from_starter.py --check <sha>` |

One process serves one service: settings and the SQL engine are process-global,
so anything that drives more than one service must give each its own process.
