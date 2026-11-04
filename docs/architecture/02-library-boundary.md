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
    "pydantic[email]>=2.12.5,<3",
    "pydantic-settings>=2.13.1,<3",
    "httpx[http2]>=0.28.1,<0.29",
    "requests>=2.32.5,<3",
    "pyhumps>=3.8.0,<4",
    "colorlog>=6.10.1,<7",
]

[project.optional-dependencies]
server   = ["fastapi>=0.133.1,<0.134", "uvicorn[standard]>=0.41.0,<0.42",
            "gunicorn>=25.1.0,<26"]
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

- **`fastapi` is an extra, not core.** A queue worker serves no HTTP, so it has no reason
  to install FastAPI and Starlette. It sits in `server` with the ASGI runners, and the
  HTTP layer — `base.app`, `base.capabilities`, `base.middleware`, `base.tools.routers`,
  `base.tools.exception_handlers`, `base.services.auth.bearer` — guards behind it.
- **`sse-starlette` is an extra, not core.** Once `main.py` moved out, nothing in `base`
  imported it; only the service's SSE endpoints do.
- **`requests` stays in core** — `HttpRequestService` uses it alongside `httpx`.
- **`aiohttp` and `certifi` were removed entirely.** Nothing imported either.
- **No `otel` extra yet.** There is no instrumentation code to install it for; it lands
  with the correlation-ID work in phase 6.

Services then declare intent legibly:

```toml
# chat_api          → base[server,sql,postgres,auth,llm,redis,sse]
# identity          → base[server,sql,postgres,auth,redis]
# ingestion_worker  → base[sql,postgres,redis]   # no server extra
```

`sqlite` is absent from all three, because all three run PostgreSQL in production and use
SQLite only as the local default (`SQL_BACKEND=sqlite`). For them it is a `dev` dependency
group: images build with `uv sync --no-dev`, so the driver stays out.

That is a per-service decision, not a rule. SQLite is a legitimate production backend —
single-replica services, read-heavy workloads, edge deployments — and a service that runs
it declares `base[sql,sqlite]` in `dependencies` like any other backend, and ships the
driver. The invariant is *declare what you ship*, not *never ship SQLite*.

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
  `sqlalchemy` and `fastapi` hard dependencies of everything. Instead the `DeclarativeBase`
  and `UploadFile` encoders are registered only when the `sql` and `server` extras are
  present; a service with neither has no declarative models and receives no uploads.
- **`base/models/definitions/errors.py` uses `http.HTTPStatus`**, not `fastapi.status`.
  The error enums are shared with non-HTTP code, and the stdlib constants are the same
  integers.
- **SQL drivers are loaded by URL, not by import**, so a missing `postgres`/`mysql`/`sqlite`
  extra would surface as a SQLAlchemy driver error rather than a `MissingExtraError`.
  `SQLClient` therefore probes for the selected backend's drivers up front and raises
  `UnsupportedBackendError`, which names the backends this build *does* support as well as
  the extra to add — the backend is chosen at runtime, so the remedy may be to change
  configuration rather than the manifest. `python -m base.preflight` runs the same
  check at container start ([doc 06](06-build-and-deploy.md)).

---

## Outbound credentials

`base` could already authenticate a **caller**. It could not authenticate **itself**. Every
service calling an OAuth2-protected dependency therefore hand-rolled token acquisition,
caching and refresh — and then the next service rebuilt it.

Applying the rule at the top of this document: would a service knowing nothing about the
consuming project still want this? Yes, in the same way it wants a SQL client. So
`base/clients/credentials/` ships alongside `clients/blob`, `clients/nosql` and
`clients/vectors` — it is named under `clients/` because this is how a service talks *out*.

```
base/clients/credentials/
  __init__.py   TokenProvider, StaticTokenProvider, BearerTokenAuth
  oauth2.py     OAuth2ClientCredentials
```

`TokenProvider` is the entire contract: one coroutine, `aget_token() -> AccessToken`. Any
credential source satisfies it — a pasted development token, a client-credentials grant, a
platform-injected managed identity.

`BearerTokenAuth` is an `httpx.Auth` that asks the provider per request. The provider serves
from its cache and refreshes when stale, so expiry becomes invisible at the call site: no
403 to catch, no retry-once, no mutating a shared client. It is async-only and its sync flow
raises, because sending the request unauthenticated would be the worse failure.

`OAuth2ClientCredentials` is plain RFC 6749, parameterised by `token_url`, `scope`,
`client_id`, `client_secret`, `credential_placement` (Basic header or form body — providers
genuinely differ), `extra_form_fields` and `refresh_skew_seconds`. It caches on the
`expires_in` the authorization server reports rather than a TTL this library invents, and
guards concurrent refresh so a burst mints one token, not N.

> **No presets module.** Those parameters are the entire difference between Azure AD, Okta,
> Auth0, Keycloak and Cognito. A named constructor for a specific provider is two lines in
> the consuming service; vendor knowledge in `base` buys nothing and dates badly. If one
> provider later proves repetitive across several services *here*, promoting a constructor
> then is easy and evidence-backed.

### Why this is not `base/services/auth/`

The names are close and the overlap is smaller than it looks. `services/auth/` is the
**inbound** half, and the two never merge because their lifecycles differ:

| | Inbound — `services/auth/` | Outbound — `clients/credentials/` |
| --- | --- | --- |
| What it does | *Verifies* a token minted elsewhere | *Mints* and caches a token |
| On whose behalf | The user who called us | The service itself |
| Lifetime | One request | Shared until it expires |
| Failure | 401 to the caller | 502 — a dependency we cannot reach |

One abstraction over both would serve neither: a verifier has no refresh, and a provider has
no principal. Concretely, `bearer.py` resolves a FastAPI caller, `tokens.py` signs RS256 and
verifies against JWKS — none of which an outbound credential does or wants.

Exactly two things are shared rather than duplicated, and both live in
`base/models/view_models/tokens.py` rather than in either half:

- **`AccessToken`**, the `NewType`. It is the same thing in both directions and is not
  redefined. It sits outside `services/auth/` deliberately: importing anything from that
  package initialises it, and it guards on `pyjwt` via the `auth` extra. A service that
  only authenticates *itself* signs and verifies nothing and must not need a JWT library
  to say so — a coupling that is invisible locally, where everything is installed, and
  surfaces as a `MissingExtraError` on import inside a per-service image.
- **`TokenResponse`** — the RFC 6749 §5.1 envelope. The inbound authorization-code
  exchange and the outbound client-credentials grant both post a form to a token URL and
  read the same reply, so both parse into this model instead of indexing into two private
  dicts. `aexchange_code` previously returned a raw dict; it now returns `TokenResponse`,
  which also means a provider answering `200` with an error body fails at the exchange
  rather than further down in profile fetching.

`services/auth/models.py` re-exports both, so `from base.services.auth import AccessToken`
keeps working for the inbound half. `Principal` stays there: it is the verified *caller*,
which only inbound has.

The token-endpoint POST itself is **not** shared. The two requests differ in grant type,
credential placement and error mapping, and a helper spanning `services/auth/` and
`clients/credentials/` would couple the inbound half to the outbound one for three lines of
`httpx` call. Each keeps its own.

### Reaching every outbound call

Two transport seams carry the hook, and both were narrower than they needed to be:

- `HttpRequestService.call_endpoint` / `acall_endpoint` typed `auth` as a `tuple[str, str]`
  basic-auth pair. Both now accept `tuple[str, str] | httpx.Auth`. The tuple path is
  unchanged; the sync path rejects an `httpx.Auth` with a message naming `acall_endpoint`,
  because `requests` cannot honour one and silently dropping it would surface as a 401.
- `openai_clients.create_clients` accepts `auth` and `verify`, applied to the httpx client
  underneath the OpenAI SDK. `OpenAIService` forwards both.

> **No credential setter on `OpenAIService`.** One was considered and rejected: it leaves
> every caller responsible for noticing expiry and retrying. The auth hook deletes that
> work rather than relocating it.

### The capability, and why `replace` is the right lever

`base/capabilities/credentials.py` keeps one credential valid and probes it. It owns the
credential and nothing else — `OutboundCredentials.auth` is a plain `httpx.Auth`, so it
works anywhere httpx does:

```python
credentials = OutboundCredentials()
app = create_app(capabilities=[credentials, ...])

await HttpRequestService().acall_endpoint(url, auth=credentials.auth)
```

`aprobe` acquires a token. A wrong client secret therefore fails `/readyz` at startup
instead of surfacing inside the first user request as an upstream 401.

An LLM endpoint behind an expiring token is one *consumer* of that credential, and lives
separately in `base/capabilities/llm.py`:

```python
capabilities = [credentials, AuthenticatedLLM(credentials.auth)]
```

`AuthenticatedLLM` constructs an `OpenAIService` carrying the hook and installs it under the
key a no-argument `OpenAIService()` resolves to:

```python
OpenAIService.replace(hash_args(), configured_service)
```

`AbstractAgent.__init__` calls `OpenAIService()` with no arguments, which `KeyedSingleton`
resolves through `hash_args()` over empty args — deterministic and stable. So every agent
picks up the configured service with no agent-side configuration, no constructor override,
and no change to `AbstractAgent`. `KeyedSingleton.replace(key, instance)` already existed
for exactly this and had no caller until now.

Two ordering facts make the swap safe, and both are commented at the call site: capabilities
start in the order listed, before the first request is served, and agents are `SingletonABC`
instances built lazily inside routers. A service that constructed an agent at *import* time
would lose the race and keep the unauthenticated client.

`ModelConfigurations` is hashed to form the `KeyedSingleton` key, so a live `httpx.Auth`
cannot ride along on it — the configured service is constructed with an explicit
`_key` instead.

Splitting the two is what keeps `base` honest. A credential that refreshes is useful to any
protected dependency — a REST API, an object store, a vendor SDK taking a prepared client —
and only one of today's consumers happens to be an LLM.

### Agent LLM surfaces

`AbstractAgent` exposed only `_aresponses`, while `OpenAIService` offers completions,
responses, and parsed variants of both. An agent needing structured output — or Chat
Completions, the only surface some OpenAI-compatible gateways expose — had to reach past its
base class. `_acompletions`, `_aparse_completions` and `_aparse_responses` now sit beside
`_aresponses` with the same `MAX_OUTPUT_TOKENS` / `REASONING_EFFORT` defaults and the same
`label` convention. The Chat Completions pair maps those ClassVars onto the names that
surface takes (`max_completion_tokens`, a flat `reasoning_effort`); the parsed variants take
no stream iterator, because a parsed call returns one complete object rather than deltas.
