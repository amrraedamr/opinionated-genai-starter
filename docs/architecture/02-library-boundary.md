# 02 — Library Boundary

The single most important question: **what belongs in `base`, and what belongs in a
service?**

## The rule

> `base` may know about *mechanisms*. It must not know about *your domain*.
>
> A shared library that ships a `users` table is not a shared library — it is a monolith
> that every service is forced to inherit.

Concretely: `base` ships abstractions, clients, and infrastructure services. It ships
**zero concrete database tables**, **zero routers**, and **zero business logic**.

---

## Current classification

Derived from which modules import `base.models.orm` (the domain marker).

### Stays in `base` — pure infrastructure

| Module | Note |
| --- | --- |
| `clients/{sql,nosql,redis_client,blob,vectors}` | Connection management. `sql/client.py` must drop its `_import_all_tables()` call |
| `services/{llm,text_embedding,vector_search,cache,blob,nosql,sql}` | Backend integrations |
| `services/{logging,execution_context,http,keyvault,common}` | Cross-cutting |
| `services/prompt` | Prompt loading/templating — mechanism, not domain |
| `services/agents` | `AbstractAgent` contract |
| `services/auth` | `TokenIssuer`, `TokenVerifier`, `Principal`, `http_bearer`, OAuth provider mechanics — authentication with **no user table**. See [doc 05](05-service-contracts.md) |
| `services/store/__init__.py` | `AbstractStore` + `AbstractSequenceStore` — **already DB-agnostic** |
| `services/store/chat_history/memory.py` | Reference in-memory implementation |
| `services/background_tasks` | Needs a distributed implementation — see [doc 07](07-modularity-backlog.md) |
| `tools/*` | Utilities. `serializers.py` imports only `Base`, which stays |
| `models/definitions`, `models/view_models` | Generic types, errors, pagination |

### Moves out of `base` — domain-coupled

| Module | Destination | Why |
| --- | --- | --- |
| `models/orm/user.py`, `user_identity.py` | `services/identity/` | Concrete tables owned by one service |
| `models/orm/conversation.py`, `chat_history_message.py` | `services/chat_api/` | Concrete tables owned by one service |
| `models/orm/prompt.py` | Whichever service owns prompt storage | Concrete table (two: `prompt` and `active_prompt`) |
| `models/definitions/database.py::DatabaseTable` | Owning service | Table *names* are part of a service's schema, not shared infrastructure |
| `services/store/{user,user_identity}.py` | `services/identity/` | Bound to those tables |
| `services/store/conversation.py`, `store/chat_history/sql.py` | `services/chat_api/` | Bound to those tables |
| `services/auth/` (the user-lookup half) | `services/identity/` | See [doc 05](05-service-contracts.md) |
| `services/prompt/templates/` | Each service | `base` ships the loading mechanism; prompts are content |
| `services/agents/test.py` | `services/chat_api/` | A concrete agent is application code; `AbstractAgent` stays |
| `models/view_models/{auth,user,conversation,chat}.py` | Owning service | Request/response models for the routers that move with them |
| `routers/{auth,conversation,user}.py` | `identity` / `chat_api` | Application surface |

> **Sequencing note.** `services/auth/` cannot be split at the same time its tables move:
> it imports `UserDTO` and `UserStore` directly. Phase 2 therefore moves **all** of
> `services/auth/` into `chat_api` so `base` is domain-free immediately, and phase 5
> extracts the stateless verification half back into `base` as part of the identity split.
> `models/view_models/chat_history.py` stays in `base` — `StoredMessage`, `ConversationId`
> and friends are generic LLM-chat types that `ChatHistoryService` and `AbstractAgent`
> depend on, not this application's domain.

### Splits in two

| Module | Shared half | Service half |
| --- | --- | --- |
| `services/auth` | Stateless token verification (JWT/JWKS, no DB) | Login, OAuth, refresh, user lookup |
| `models/orm/__init__.py` | `TimestampMixin`, `to_dict`/`clone`/`update_from_dict` helpers | Each service's own `DeclarativeBase` and tables |

---

## Three side effects that must be removed

These are load-bearing today and will silently break per-service isolation.
### 1. Automatic table import

```python
# base/models/orm/__init__.py — current, at module bottom
def _import_all_tables():
    for _, name, _ in pkgutil.walk_packages(__path__):
        importlib.import_module(f"base.models.orm.{name}")

_import_all_tables()        # runs on import
```

Also invoked from `base/clients/sql/client.py`. Consequence: **any** service touching SQL
registers **every** table into the shared `Base.metadata`, and `alembic autogenerate`
would try to create all of them in that service's database.

**Replacement:** each service's `alembic/env.py` imports its own models explicitly. Explicit
beats magic exactly when isolation matters.

```python
# services/chat_api/alembic/env.py
import chat_api.models.orm.chat_history_message    # noqa: F401
import chat_api.models.orm.conversation            # noqa: F401
import chat_api.models.orm.prompt                  # noqa: F401
from base.tools.migrations import run_migrations
from chat_api.models.orm import ChatBase
from chat_api.settings import settings

run_migrations(ChatBase.metadata, settings)
```

### 2. Import-time application construction

```python
# base/main.py — current
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
init_routers(app, routers=routers, prefix=settings.api_prefix)
AppExceptionHandler().register(app)
```

Everything happens at import time with hardcoded routers, a hardcoded prefix, and a
hardcoded wide-open CORS policy. A second service cannot reuse any of it.

### 3. Hardcoded prompt templates and a hardcoded chat-history store

Two more "the library knows which application it is" couplings, found while doing the
extraction:

```python
# base/services/prompt/loader.py — current
import base.services.prompt.templates as pkg
...
for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
```

```python
# base/services/chat_history/__init__.py — current
self._store: SqlChatHistoryStore = SqlChatHistoryStore()
```

The first makes `base` the only place prompts can live; the second binds a `base` service
to a concrete table, contradicting the `ChatHistoryStore` docstring that already says
*"ChatHistoryService depends on this type, not on AbstractSequenceStore directly"*.

**Replacement:** `PromptLoader.initialize(template_packages)` takes the service's own
packages, and the templates move out of `base`. `ChatHistoryService(config, store)` takes
the store as a **type**, defaulting to the in-memory reference implementation that stays in
`base`. A type rather than an instance because `ChatHistoryService` is a `KeyedSingleton`
whose identity key is `hash_args(...)` over its constructor arguments — a store *instance*
has no serializer and would raise; a class is handled by the existing `encode_type`.

`AbstractAgent` gains a `CHAT_HISTORY_STORE` class variable for the same reason, defaulting
to in-memory. A service must point its agents and its routers at the *same* store class, or
the two resolve to different `ChatHistoryService` instances and read different history.

---

## The app factory

`base` provides the composition machinery; each service calls it.

```python
# packages/base/base/app.py
def create_app(
    *,
    settings: CoreSettings,
    routers: ModuleType | None = None,
    prefix: str = "/api",
    capabilities: Sequence[Capability] = (),
    lifespan_hooks: Sequence[LifespanHook] = (),
    cors_origins: Sequence[str] | None = None,
    title: str = DEFAULT_TITLE,
    version: str = DEFAULT_VERSION,
) -> FastAPI:
    ...
```

Responsibilities:

- build `FastAPI` with a lifespan that runs **only** the registered hooks
- install correlation-ID middleware (see [doc 05](05-service-contracts.md))
- install CORS from settings — never `["*"]` by default outside local
- `init_routers(...)` on the service's router package (reuse as-is; it already works)
- register `AppExceptionHandler`
- mount `/livez` and `/readyz` backed by the capability probes

A service composition root then looks like this:

```python
# services/chat_api/chat_api/app.py
from base.app import create_app
from base.capabilities.prompts import Prompts
from base.capabilities.terminal_capture import TerminalCapture
from chat_api import routers
from chat_api.prompts import templates
from chat_api.settings import settings

app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[
        TerminalCapture(settings.service_name),
        Prompts(templates=[templates]),
    ],
    cors_origins=settings.cors_origins_list,
)


def start() -> None:
    from base.tools.webserver import run_webserver
    run_webserver(app_path="chat_api.app:app", settings=settings)
```

Connection-backed capabilities (`SQL()`, `Redis()`, `LLM()`) that make `/readyz` probe real
infrastructure are a natural next addition — see [doc 07](07-modularity-backlog.md).

### Capabilities replace the hardcoded lifespan

Today's lifespan unconditionally starts `PromptLoader` and `TerminalCaptureService`. A
worker service with no prompts should not pay for either.

A `Capability` is a small object declaring what it needs at startup, what to tear down, and
how to report readiness:

```python
class Capability(Protocol):
    name: str
    async def astartup(self, app: FastAPI) -> None: ...
    async def ashutdown(self, app: FastAPI) -> None: ...
    async def aprobe(self) -> ProbeResult: ...
```

Capabilities start in the order listed and shut down in reverse. A capability that fails to
start does not strand the ones already running, and an `ashutdown()` that raises does not stop
the rest from running.

> **Instantiate, do not connect.** A capability object is built at module import, and
> gunicorn runs with `preload: True` — so anything created in `__init__` is created *before*
> `fork()` and shared by every worker. Connection pools, engines and clients are not
> fork-safe. Build them in `astartup()`, which the ASGI lifespan runs once per worker after
> the fork.

This also yields a real `/readyz`: each enabled capability reports whether its backing
dependency is reachable, instead of today's static `{"status": "ok"}`. Probes are gathered
concurrently, a raising probe is reported as `failed` rather than bubbling out, and the
worst individual status becomes the overall one — `failed` answers 503, everything else
answers 200. `/livez` deliberately touches nothing: it reports only that the process is up,
so a transient dependency outage does not get the container killed.

### Webserver must stop hardcoding the app path

```python
# base/tools/webserver.py — current
uvicorn.run("base.main:app", ...)
app = importlib.import_module("base.main").app
```

Both need `app_path` as a parameter. Note also that gunicorn is configured with
`preload: True`; combined with singletons created at import time, any connection pool
created before `fork()` is **not** fork-safe. Each service must create engines and
clients inside a lifespan hook, not at module import. Worth an explicit test.

---

## Dependency extras

`base` currently declares ~40 hard dependencies. A worker that only needs Redis and an LLM
would still install `qdrant-client`, `azure-search-documents`, `azure-cosmos`, `pymongo`,
`asyncpg`, `psycopg2-binary`, `aiomysql`, `pymysql` and `tiktoken`.

Because `base/services/__init__.py` and `base/clients/__init__.py` are **empty**, imports
are already leaf-level — so splitting into extras requires no import refactor.

```toml
# packages/base/pyproject.toml
[project]
dependencies = [
    "fastapi>=0.133.1,<0.134",
    "pydantic[email]>=2.12.5,<3",
    "pydantic-settings>=2.13.1,<3",
    "httpx[http2]>=0.28.1,<0.29",
    "requests>=2.32.5,<3",
    "pyhumps>=3.8.0,<4",
    "colorlog>=6.10.1,<7",
]

[project.optional-dependencies]
server   = ["uvicorn[standard]>=0.41.0,<0.42", "gunicorn>=25.1.0,<26"]
sql      = ["sqlalchemy>=2.0.47,<3", "alembic>=1.18.4,<2",
            "sqlalchemy-utils>=0.42.1,<0.43", "greenlet>=3.3.2,<4"]
sqlite   = ["aiosqlite>=0.22.1,<0.23"]
postgres = ["asyncpg>=0.31.0,<0.32", "psycopg2-binary>=2.9.11,<3"]
mysql    = ["aiomysql>=0.3.2,<0.4", "pymysql>=1.1.2,<2"]
auth     = ["pyjwt>=2.11.0,<3", "argon2-cffi>=25.1.0,<26", "cryptography>=46.0.5,<47"]
llm      = ["openai>=2.24.0,<3", "tiktoken>=0.12.0,<0.13"]
redis    = ["redis>=7.2.1,<8"]
mongo    = ["pymongo>=4.16.0,<5"]
vectors  = ["qdrant-client>=1.17.0,<2"]
sse      = ["sse-starlette>=3.2.0,<4"]
azure    = ["azure-identity>=1.25.2,<2", "azure-keyvault-secrets>=4.10.0,<5",
            "azure-storage-blob>=12.28.0,<13", "azure-cosmos>=4.15.0,<5",
            "azure-search-documents>=11.6.0,<12"]
```

Notes from doing it:

- **`sse-starlette` is an extra, not core.** Once `main.py` moved out, nothing in `base`
  imported it; only the service's SSE endpoints do.
- **`requests` stays in core** — `HttpRequestService` uses it alongside `httpx`.
- **`aiohttp` and `certifi` were removed entirely.** Nothing imported either.
- **No `otel` extra yet.** There is no instrumentation code to install it for; it lands
  with the correlation-ID work in phase 6.

Services then declare intent legibly:

```toml
# chat_api          → base[server,sql,sqlite,postgres,auth,llm,redis,sse]
# identity          → base[server,sql,postgres,auth,redis]
# ingestion-worker  → base[sql,postgres,llm,vectors,mongo]   # no server extra
```

Measured, not assumed: with `chat_api` installed into its own environment, its
`site-packages` contains `asyncpg, gunicorn, httpx, openai, redis, sqlalchemy, tiktoken,
uvicorn` and **no** `qdrant_client`, `pymongo`, `azure*` or `aiomysql`. A probe service
declaring `base[vectors,mongo]` gets the exact inverse — `httpx, pymongo, qdrant_client`
and no `openai`, `redis`, `sqlalchemy` or `gunicorn` — with both resolving to the identical
`httpx` build from the single root lockfile.

### Guarding a missing extra

An extras split needs friendly failure. Each optional subsystem imports behind a guard:

```python
try:
    from qdrant_client import AsyncQdrantClient
except ModuleNotFoundError as exc:   # pragma: no cover
    raise MissingExtraError("vectors", "qdrant-client") from exc
```

`MissingExtraError` renders as: *"QdrantClient requires the 'vectors' extra. Add
`base[vectors]` to this service's pyproject.toml (installs qdrant-client)."* Ruff's `I`
ruleset is already enabled, so import hygiene stays enforced.

Two refinements the implementation needed:

- **`base/tools/serializers.py` degrades instead of raising.** It is reachable from
  `hash_args` → `KeyedSingleton`, so it is effectively core. Raising there would make
  `sqlalchemy` a hard dependency of everything. Instead the `DeclarativeBase` encoder is
  registered only when the `sql` extra is present; a service with no SQLAlchemy has no
  declarative models to encode anyway.
- **SQL drivers are loaded by URL, not by import**, so a missing `postgres`/`mysql`/`sqlite`
  extra would surface as a SQLAlchemy driver error rather than a `MissingExtraError`.
  `SQLClient` therefore probes for the selected backend's drivers up front and raises the
  named error itself.
