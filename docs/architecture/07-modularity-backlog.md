# 07 — Modularity Backlog

> *"What else would make this more modular for microservices/apps?"*

Docs 01–06 cover the structural changes. This document lists everything else found while
reading the codebase, ranked by impact on modularity.

| # | Item | Impact | Effort | State today |
| --- | --- | --- | --- | --- |
| 1 | Service scaffolding template | High | Medium | None |
| 2 | Distributed task broker | High | Medium | In-memory only |
| 3 | Capability registry → real `/readyz` | High | Low | Hardcoded lifespan, static health |
| 4 | Correlation ID propagation | High | Low | Generated, never propagated |
| 5 | Boundary lint in CI | High | Low | None |
| 6 | Test infrastructure | High | Medium | **No tests at all** |
| 7 | Resource namespacing discipline | Medium | Low | Partial (`CacheChannels.for_app`) |
| 8 | Fork-safety of singletons | Medium | Low | `preload=True` + import-time singletons |
| 9 | Error code namespacing | Medium | Low | Global enums |
| 10 | OpenTelemetry | Medium | Low | None |
| 11 | `base` versioning escape hatch | Medium | Low | N/A |
| 12 | Aggregated API documentation | Low | Low | Per-app `/docs` |
| 13 | Developer task runner | Low | Low | `run.sh` only |

---

## 1. Service scaffolding template

With one app, consistency is free. With six, it decays — service four invents a different
settings layout, service five skips the readiness probe.

A **starter should generate services, not just host them.**

```bash
uv run python tools/new_service.py \
    --name billing \
    --storage postgres \      # postgres | mysql | mongo | none
    --http \                  # omit for a worker
    --capabilities llm,redis
```

Generates `pyproject.toml` with the right `base[...]` extras, `app.py` calling
`create_app()`, `settings.py` with the right mixins, `.env.example`, `Dockerfile`,
`alembic/` (only when SQL), a `routers/` package (only when `--http`), and a test skeleton.

This is arguably the **highest-leverage item in the list**: it turns every decision in these
documents into the default outcome rather than something each developer has to re-read.
Use `copier` so existing services can also be *updated* when the template evolves.

---

## 2. Distributed task broker

`base/services/background_tasks/` contains only `in_memory.py`. In-process tasks cannot
cross a process boundary and are lost on restart — so there is currently no way for
services to communicate asynchronously.

`AbstractBroker` with in-memory / Redis Streams / Service Bus implementations, designed as
a sibling of `AbstractStore`. Full sketch in [doc 05](05-service-contracts.md).

This also unlocks the **worker service** shape: no HTTP server, no `server` extra, just a
subscription loop — a category of service the current structure cannot express at all.

---

## 3. Capability registry and a real `/readyz`

Today's lifespan unconditionally starts `PromptLoader` and `TerminalCaptureService`, and
`/health-check` returns a static `{"status": "ok"}` — it reports healthy while the database
is unreachable.

Capabilities ([doc 02](02-library-boundary.md)) fix both: a service enables exactly the
subsystems it uses, and each contributes a readiness probe. Split into `/livez` (restart
signal) and `/readyz` (load-balancer signal).

Low effort, and it removes the main reason a worker service would otherwise be forced to
carry prompt-loading machinery.

---

## 4. Correlation ID propagation

`base/main.py` sets `scope_id.set(uuid4())` per request but never reads an inbound header,
so one logical request crossing three services produces three unrelated IDs.

Adopt inbound `X-Correlation-ID` / W3C `traceparent`, propagate through `ServiceClient` and
broker messages, and include `correlation_id` + `service_name` in every log line. Details
in [doc 05](05-service-contracts.md).

Small change; it is the difference between debugging a distributed system and guessing.

---

## 5. Boundary lint in CI

Every rule in these documents is enforceable mechanically:

| Rule | Check |
| --- | --- |
| No service imports another service | Import graph scan |
| `base` imports no service | Import graph scan |
| `base` declares no `__tablename__` | AST scan of `packages/base` |
| No `ForeignKey` crosses a service | AST scan of each service's ORM |
| Each service has exactly one `DeclarativeBase` | AST scan |
| No `.env` or key material in any image | Scan built images |

Architecture that is not enforced becomes architecture that used to be true. A ~100-line
script in CI keeps these documents accurate.

---

## 6. Test infrastructure

There are currently **no tests anywhere in the repository** — no `tests/` directory, no
`pytest` dependency, no fixtures.

For a single app that is a gap. For a shared library consumed by N services it is a
liability: there is nothing to tell you that a `base` change broke service four.

| Layer | Scope |
| --- | --- |
| `packages/base` | Unit tests for abstractions. `AbstractStore` conformance suite reusable by any implementation |
| Per service | Unit + API tests, `testcontainers` for a real database |
| Contract tests | Consumer-driven, per service pair, run on both sides in CI |
| Smoke | `docker compose up` + health checks across the stack |

The `AbstractStore` conformance suite is the valuable one: a single parameterised test
module that any store implementation — SQL, Mongo, in-memory — must pass. It makes
"swap the backend" a verified claim rather than an aspiration.

Pairs with item 5: the affected-service matrix in [doc 06](06-build-and-deploy.md) is only
useful once there is something to run.

---

## 7. Resource namespacing discipline

`CacheChannels.for_app(app_name)` already prefixes Redis pub/sub per app. Generalise to
**every** shared resource, so services sharing infrastructure cannot collide:

| Resource | Convention |
| --- | --- |
| Redis channels | `{service}:{channel}` — already done |
| Cache keys | `{service}:{entity}:{id}` |
| Blob containers | `{service}-{purpose}` |
| Queue topics | `{service}.{event}` |
| Vector collections | `{service}_{collection}` |
| Log fields | Always include `service_name` |

Provide a single `Namespace(service_name)` helper in `base` so this is one call rather than
a convention people remember.

---

## 8. Fork-safety of singletons

`base/tools/webserver.py` configures gunicorn with `preload: True`, and `Singleton`
instances are created on first use — which, for anything touched at import time, is
*before* `fork()`.

Connection pools (SQLAlchemy async engines, Redis, Mongo clients) created pre-fork are
shared across workers and produce intermittent, hard-to-reproduce failures.

Rule: **create no client at import time.** Capabilities create their clients in `astartup()`,
which runs post-fork. Worth a dedicated test that asserts no engine exists after importing
a service module.

This matters more with many services because each new service re-inherits the pattern.

---

## 9. Error code namespacing

`base/models/definitions/errors.py` defines global enums (`DatabaseErrors`, etc.) and
`tools/locale.py` + `canned_messages` localise them. With several services, a gateway or
client cannot tell which service produced `RECORD_NOT_FOUND`.

Prefix error codes with the owning service (`chat_api.RECORD_NOT_FOUND`) and keep
per-service message catalogues, with `base` owning only infrastructure-level codes. Keep
the existing `AppExceptionHandler` shape — only the code namespace changes.

---

## 10. OpenTelemetry

An optional `base[otel]` extra that auto-instruments FastAPI, httpx and SQLAlchemy, exports
OTLP, and binds `service.name` to `CoreSettings.service_name`. Off by default; one setting
to enable.

The existing `debug_trace` view models and `execution_context` service already model
execution spans — OTel export complements rather than replaces them.

---

## 11. `base` versioning escape hatch

In a workspace, `base = { workspace = true }` means every service tracks `base` at HEAD.
That is correct by default and is why the single lockfile is valuable.

But it makes `base` changes globally breaking, so document the escape hatch:

- **Default:** workspace reference. A breaking `base` change updates all consumers in the
  same PR.
- **Escape hatch:** publish `base` to a private index and let a lagging service pin
  `base==1.4.0` temporarily.
- **Deprecation policy:** `base` keeps a shim for one minor version with a
  `DeprecationWarning` before removing a public API.

Without a stated policy, the first painful cross-service change turns into a copy-paste of
`base`.

---

## 12. Aggregated API documentation

Each service serves its own `/docs`. A gateway can merge the OpenAPI schemas — or CI can
publish a static combined document — so consumers see one API surface. Low effort once
prefixes come from settings ([doc 02](02-library-boundary.md)).

---

## 13. Developer task runner

`run.sh` is a container entrypoint, not a developer interface. With N services, add a
`Makefile` or `Taskfile.yml`:

```
make up             # infra + all services
make dev s=chat_api # infra in docker, one service on host with reload
make migrate s=chat_api
make test s=chat_api
make new s=billing
```

Cosmetic individually; collectively it is what stops people shelling out to half-remembered
`docker compose` incantations.

It must export `UV_PROJECT_ENVIRONMENT=.venvs/$(s)` for per-service targets — a uv
workspace otherwise shares one `.venv`, and `uv sync --package b` uninstalls service `a`.
See [doc 01](01-workspace-layout.md).

---

## 14. Connection-backed readiness capabilities

Phase 1 landed the `Capability` protocol and `/readyz`, but only `Prompts` and
`TerminalCapture` implement it. Neither touches a network. A service whose database or
cache is unreachable still reports ready.

Add `SQL()`, `Redis()`, `LLM()` and `Vectors()` capabilities whose `probe()` issues a
cheap round trip (`SELECT 1`, `PING`, a models list) and whose `shutdown()` disposes the
pool. They must create their clients in `astartup()`, never in `__init__` — gunicorn runs
with `preload: True`, so an `__init__`-built pool is shared across forked workers.

---

## 15. Per-service prompt template discovery

`PromptLoader.initialize(template_packages)` now takes the packages explicitly, and
templates live in the service (`chat_api/prompts/templates/`). What is still missing is
any check that two services' prompt *keys* do not collide when they share a Redis
`PROMPT_RELOAD` channel. `CacheChannels.for_app(...)` already namespaces the channel;
the registry inside `PromptLoader` is still flat and global.

---

## Deliberately excluded

| Not doing | Why |
| --- | --- |
| Service mesh | Base URLs in settings cover discovery. Revisit at ~10 services |
| Distributed transactions / sagas | Eventual consistency plus idempotent handlers is sufficient here |
| Shared "common domain" package | Becomes a dumping ground and recreates the coupling being removed. `contracts` is deliberately narrow |
| gRPC | HTTP + Pydantic matches the existing stack. Revisit only if latency demands it |
| Per-service repositories | Loses the atomic cross-service refactor that justifies the workspace |
