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
| `services/store/__init__.py` | `AbstractStore` + `AbstractSequenceStore` — **already DB-agnostic** |
| `services/store/chat_history/memory.py` | Reference in-memory implementation |
| `services/background_tasks` | Needs a distributed implementation — see [doc 07](07-modularity-backlog.md) |
| `tools/*` | Utilities. `serializers.py` imports only `Base`, which stays |
| `models/definitions`, `models/view_models` | Generic types, errors, pagination |

### Moves out of `base` — domain-coupled

| Module | Destination | Why |
| --- | --- | --- |
| `models/orm/user.py`, `user_identity.py` | `services/identity/` | Concrete tables owned by one service |
| `models/orm/conversation.py`, `chat_history_message.py` | `services/chat-api/` | Concrete tables owned by one service |
| `models/orm/prompt.py` | Whichever service owns prompt storage | Defines **two** tables: `prompt` and `active_prompt` — they must move together |
| `services/store/{user,user_identity}.py` | `services/identity/` | Bound to those tables |
| `services/store/conversation.py`, `store/chat_history/sql.py` | `services/chat-api/` | Bound to those tables |
| `services/auth/` (the user-lookup half) | `services/identity/` | See [doc 05](05-service-contracts.md) |
| `routers/{auth,conversation,user}.py` | `identity` / `chat-api` | Application surface |

### Splits in two

| Module | Shared half | Service half |
| --- | --- | --- |
| `services/auth` | Stateless token verification (JWT/JWKS, no DB) | Login, OAuth, refresh, user lookup |
| `models/orm/__init__.py` | `TimestampMixin`, `to_dict`/`clone`/`update_from_dict` helpers | Each service's own `DeclarativeBase` and tables |

---

## Two side effects that must be removed

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
# services/chat-api/alembic/env.py
from chat_api.models.orm import ChatBase
import chat_api.models.orm.conversation            # noqa: F401
import chat_api.models.orm.chat_history_message    # noqa: F401

target_metadata = ChatBase.metadata
```

### 2. Import-time application construction

```python
# base/main.py — current
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
init_routers(app, routers=routers, prefix="/api/7r")
AppExceptionHandler().register(app)
```

Everything happens at import time with hardcoded routers, a hardcoded prefix, and a
hardcoded wide-open CORS policy. A second service cannot reuse any of it.

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
# services/chat-api/chat_api/app.py
from base.app import create_app
from base.capabilities import LLM, Redis, SQL
from chat_api import routers
from chat_api.settings import settings

app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[SQL(), Redis(), LLM()],
    cors_origins=settings.cors_origins_list,
)


def start() -> None:
    from base.tools.webserver import run_webserver
    run_webserver(app_path="chat_api.app:app", settings=settings)
```

### Capabilities replace the hardcoded lifespan

Today's lifespan unconditionally starts `PromptLoader` and `TerminalCaptureService`. A
worker service with no prompts should not pay for either.

A `Capability` is a small object declaring what it needs at startup, what to tear down, and
how to report readiness:

```python
class Capability(Protocol):
    name: str
    async def startup(self, app: FastAPI) -> None: ...
    async def shutdown(self, app: FastAPI) -> None: ...
    async def probe(self) -> ProbeResult: ...
```

This also yields a real `/readyz`: each enabled capability reports whether its backing
dependency is reachable, instead of today's static `{"status": "ok"}`.

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
    "pyhumps>=3.8.0,<4",
    "colorlog>=6.10.1,<7",
    "sse-starlette>=3.2.0,<4",
]

[project.optional-dependencies]
server   = ["uvicorn[standard]>=0.41.0,<0.42", "gunicorn>=25.1.0,<26"]
sql      = ["sqlalchemy>=2.0.47,<3", "alembic>=1.18.4,<2",
            "sqlalchemy-utils>=0.42.1,<0.43", "greenlet>=3.3.2,<4"]
postgres = ["asyncpg>=0.31.0,<0.32", "psycopg2-binary>=2.9.11,<3"]
mysql    = ["aiomysql>=0.3.2,<0.4", "pymysql>=1.1.2,<2"]
sqlite   = ["aiosqlite>=0.22.1,<0.23"]
auth     = ["pyjwt>=2.11.0,<3", "argon2-cffi>=25.1.0,<26", "cryptography>=46.0.5,<47"]
llm      = ["openai>=2.24.0,<3", "tiktoken>=0.12.0,<0.13"]
redis    = ["redis>=7.2.1,<8"]
mongo    = ["pymongo>=4.16.0,<5"]
vectors  = ["qdrant-client>=1.17.0,<2"]
azure    = ["azure-identity>=1.25.2,<2", "azure-keyvault-secrets>=4.10.0,<5",
            "azure-storage-blob>=12.28.0,<13", "azure-cosmos>=4.15.0,<5",
            "azure-search-documents>=11.6.0,<12"]
otel     = ["opentelemetry-sdk", "opentelemetry-exporter-otlp",
            "opentelemetry-instrumentation-fastapi",
            "opentelemetry-instrumentation-httpx"]
```

Services then declare intent legibly:

```toml
# identity          → base[server,sql,postgres,auth,redis]
# chat-api          → base[server,sql,postgres,llm,redis,auth]
# ingestion-worker  → base[sql,postgres,llm,vectors,mongo]   # no server extra
```

### Guarding a missing extra

An extras split needs friendly failure. Each optional subsystem imports behind a guard:

```python
try:
    from qdrant_client import AsyncQdrantClient
except ModuleNotFoundError as exc:   # pragma: no cover
    raise MissingExtraError("vectors", "qdrant-client") from exc
```

`MissingExtraError` renders as: *"VectorSearchService requires the 'vectors' extra. Add
`base[vectors]` to this service's pyproject.toml."* Ruff's `I` ruleset is already enabled,
so import hygiene stays enforced.
