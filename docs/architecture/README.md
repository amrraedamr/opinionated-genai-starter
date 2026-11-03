# Multi-Service Architecture — Design Docs

Design for evolving this starter from **one application** into a **multi-app / microservice
starter** that still shares a single opinionated foundation.

> **Status: implemented.** All seven phases described here are built. The workspace holds
> `base` and `contracts` under `packages/`, and `chat_api`, `identity` and
> `ingestion_worker` under `services/`. Where implementation diverged from the original
> design, these documents were corrected to match the code.

---

## The core insight

The repository is already ~97% reusable library and ~3% application:

| Package | Lines | Nature |
| --- | ---: | --- |
| `base/services` | 11,335 | Infrastructure — reusable |
| `base/models` | 2,108 | Mostly reusable (5 concrete tables are not) |
| `base/tools` | 1,111 | Infrastructure — reusable |
| `base/clients` | 860 | Infrastructure — reusable |
| `base/routers` | **414** | **Application surface — the only truly app-specific code** |

So this is **not a rewrite**. It is promoting `base` from *"the application"* to
*"a shared library that applications consume"*, and extracting the small app-specific
remainder into the first service.

Three existing design decisions make this unusually cheap:

1. **`AbstractStore` / `AbstractSequenceStore` are already storage-agnostic.** The
   docstring states *"Works with SQL and NoSQL services"*, and `_duplicate_error_types()`
   exists precisely to let each backend map its own driver errors. Per-service database
   freedom is already supported by the abstraction — it is only blocked by `base` also
   shipping concrete, table-bound stores.
2. **`base/services/__init__.py` and `base/clients/__init__.py` are empty.** Imports are
   leaf-level, so dependencies can be split into optional extras with no import refactor.
3. **`init_routers(app, routers=..., prefix=...)` is already parameterised**, and
   `CacheChannels.for_app(app_name)` already namespaces Redis pub/sub per application.

---

## Decisions at a glance

| # | Decision | Rationale | Doc |
| --- | --- | --- | --- |
| 1 | **uv workspace monorepo**, one lockfile, many packages | Repo already uses uv + `uv.lock`; one resolution keeps transitive versions consistent across services | [01](01-workspace-layout.md) |
| 2 | `base` becomes a **domain-free library**; it ships zero tables and zero routers | A shared library that knows about `users` is not shared, it is a monolith with extra steps | [02](02-library-boundary.md) |
| 3 | **Dependencies split into extras** — `base[sql,llm,redis,…]` | Today's 40 hard deps would force a worker service to install Qdrant, Cosmos, Azure Search, MySQL *and* Postgres | [02](02-library-boundary.md) |
| 4 | **Database per service.** Each service picks SQL, NoSQL or nothing, owns its own schema, its own migrations, its own connection | Independent deployability; the store abstraction already allows it | [03](03-data-ownership.md) |
| 5 | **Each service defines its own `DeclarativeBase`** — never a shared one | Sharing `Base` shares `metadata`, which silently couples every service's migrations | [03](03-data-ownership.md) |
| 6 | **No cross-service foreign keys.** Foreign entities are opaque IDs | FKs are a physical coupling that defeats separate databases | [03](03-data-ownership.md) |
| 7 | **`CoreSettings` + opt-in mixins**, composed per service | A single inherited god-class forces every service to carry OAuth and vector fields it never uses | [04](04-configuration.md) |
| 8 | **Each service has its own `.env`**, layered, with process env winning | Different deployments, different LLM keys, different databases | [04](04-configuration.md) |
| 9 | **Auth splits into stateless verification (shared) + identity service (owns users)** | Every service can verify a JWT with a public key; only one service should own the user table | [05](05-service-contracts.md) |
| 10 | **One parameterised Dockerfile**, `uv sync --package $SERVICE` | Per-service images without per-service build drift | [06](06-build-and-deploy.md) |

---

## Document index

| Doc | Covers |
| --- | --- |
| [01 — Workspace Layout](01-workspace-layout.md) | uv workspace, directory structure, how packages reference each other |
| [02 — Library Boundary](02-library-boundary.md) | What stays in `base`, what moves out, dependency extras, the app factory |
| [03 — Data Ownership](03-data-ownership.md) | Database per service, SQL/NoSQL freedom, metadata isolation, per-service migrations |
| [04 — Configuration](04-configuration.md) | Settings composition, per-service `.env`, precedence, secrets |
| [05 — Service Contracts](05-service-contracts.md) | Auth propagation, sync + async communication, correlation IDs, shared contracts |
| [06 — Build & Deploy](06-build-and-deploy.md) | Dockerfile template, compose, CI affected-service detection |
| [07 — Modularity Backlog](07-modularity-backlog.md) | Remaining gaps that would make this genuinely modular |
| [08 — Migration Plan](08-migration-plan.md) | Phased, incremental path — the app keeps running at every step |

---

## Non-goals

- **Not prescribing a service mesh, Kubernetes operator, or orchestrator.** The output is
  container images plus configuration; where they run stays open.
- **Not forcing every consumer into microservices.** A single-app user should still be able
  to clone, generate one service, and get today's experience. The workspace must degrade
  gracefully to one member.
- **Not introducing distributed transactions.** Cross-service consistency is eventual, via
  events and idempotent handlers.
