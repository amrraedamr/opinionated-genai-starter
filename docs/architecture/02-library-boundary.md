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

Which provider answers is a per-agent decision, not a per-application one:
`LLM_SERVICE` is a ClassVar like `CHAT_HISTORY_STORE`, so a cheap router and an
expensive answerer can sit on different providers in the same deployment.
`MODEL_CONFIGURATIONS` pins a specific model when the provider's default is not
the one that agent wants.

Leaving `MODEL_CONFIGURATIONS` at `None` matters more than it looks. An LLM
service is a `KeyedSingleton` whose key is derived from its constructor
arguments, so `Service()` and `Service(None)` resolve *different* instances. An
agent that passed an explicit `None` would miss the client `AuthenticatedLLM`
installs and would silently call the endpoint unauthenticated. The agent
therefore calls with no arguments unless a configuration is actually set.

`AbstractAgent` exposes `_acomplete` and `_aparse`, both of which go through
`AbstractLLMService`. An agent passes a `RenderedPrompt` or `StoredMessage` list and gets an
`LLMResult` back; which provider answered, and which of its API surfaces was used, is the
adapter's business. Both apply the agent's `MAX_OUTPUT_TOKENS` / `REASONING_EFFORT` defaults
and the same `label` convention, and `_aparse` takes no stream iterator because a parsed call
returns one complete object rather than deltas.

`LLMResult.finish_reason` is normalised to a `FinishReason` rather than passed
through. The providers disagree on vocabulary and even on meaning — the Responses
API reports a *status* (`completed`), Chat Completions a reason (`length`), Anthropic
a third set (`end_turn`, `tool_use`) and Gemini a fourth (`STOP`, `MAX_TOKENS`) — so a caller branching on the raw string would
work against one provider and silently never fire against another. `LENGTH` in
particular means the reply is truncated, which is worth branching on.

`max_output_tokens` and `reasoning_effort` are neutral parameters on the seam. Each adapter
spells them the way its provider wants: the Responses API takes `max_output_tokens` and a
nested `reasoning` block, Chat Completions takes `max_completion_tokens` and a flat
`reasoning_effort`, and Anthropic takes `max_tokens` with a thinking budget in tokens, as Gemini does
through `thinking_config`.
`ModelConfigurations.supports_responses_api` picks the OpenAI surface, so a gateway that only
serves Chat Completions needs no change at the call site.

### Where the LLM code lives

`AuthenticatedLLM` installs whichever service `service_type` names, defaulting to
OpenAI; it reads `AbstractLLMService` and the `KeyedSingleton` protocol, never a
provider's SDK. The credential it carries is typed by `OutboundAuth`, a structural
protocol, because the OpenAI SDK ships on `httpx` and the Anthropic SDK on `httpx2`
— two unrelated `Auth` hierarchies that reject each other at runtime. `base.clients.credentials`
therefore ships a binding per library — `BearerTokenAuth` on `httpx` in the package
itself, `Httpx2BearerTokenAuth` in `credentials/httpx2.py` because `httpx2` arrives
only with the `anthropic` extra — both over the shared `TokenProvider`, so one
acquired token serves either.

`base.services.llm` holds only what every provider shares: `AbstractLLMService`. Each
provider gets a package — `llm/openai/`, `llm/anthropic/`, `llm/gemini/` — holding its
service in `__init__.py` beside its message adapter and its error handler. Adding a
provider adds a directory rather than widening a shared module, and importing the seam
pulls in no SDK.

Gemini is where that claim was tested. Three of its differences reach the adapter rather
than the caller. Its assistant role is called `model`; its instructions are a request
field rather than a role, as Anthropic's are; and a tool result is matched to the call it
answers by **function name** rather than by an id, so the id this library carries has to
be resolved back to the name the model used before a result can be sent — and an id has
to be invented for a call coming the other way, because Gemini sends none. Its structured
output needs no such work: `response_schema` constrains decoding directly, where the
Anthropic adapter forces a tool call and unwraps it.

Its token accounting needed the same care Anthropic's did, in the opposite direction.
Gemini reports `thoughts_token_count` alongside `candidates_token_count` rather than
inside it, while the neutral shape treats reasoning as a share of the output total, so
the two are added. Cached input needs no handling at all: `prompt_token_count` already
includes it, which is what the neutral shape expects and the reverse of Anthropic, whose
cache counts sit outside the input total.

`AbstractTextEmbeddingService` is the same arrangement for embeddings, with
`text_embedding/openai/` beside it. Batching belongs to the abstraction rather than
to each adapter: splitting a list to fit a provider's per-request ceiling is the same
work whoever answers, and an adapter reimplementing it could silently return vectors
out of order. An adapter implements one batch and nothing else.

`AbstractKeyVaultService` closes the last capability that had none. A secret store was
shipped as a concrete Azure class with an empty `__init__.py`, so a service reaching for
a runtime secret would have imported the vendor, which is the coupling the seam exists
to remove — and `list_secrets` handed back Azure's own `SecretProperties`, putting the
SDK in the return type. A neutral `SecretInfo` carries what every vault knows, and lives
in `view_models/` with the other capabilities' shapes rather than in the service module.
Fetching several secrets belongs to the abstraction for the same reason batching does:
doing them one at a time is what keeps each name's retries its own, and it is the same
work whoever answers. Naming rules stay with the adapter, because what a legal secret
name looks like is the vault's rule, not the caller's.

Retrying a call that leaves this process is not an LLM concern, so it is not in the LLM
package. `base.services.error_handling` owns the loop, the jittered backoff and the
logging, and names no domain at all. The base never constructs a domain error itself —
`wrap`, `wrap_exhausted` and `wrap_unexpected` belong to the implementation — so a store
handler raises store errors and a model handler raises model errors over one retry policy.

An SDK that serves more than one capability gets its reader at the root, where it
reads exceptions without deciding what callers see: `openai.py` for generation and
embeddings, `azure.py` for object storage and search, `sqlalchemy.py` for anything
reaching a database. That is why an Azure 429 becomes a blob error in one place and
a vector-search error in another from the same classification.

Retrying a SQLAlchemy failure is only safe where the statement is idempotent and the
method opens its own transaction, because a failed session cannot be resumed. The
pgvector backend qualifies — DDL guarded by `IF NOT EXISTS`, writes upserted with
`ON CONFLICT DO UPDATE`, deletes by id.

Which failure it is gets decided on the server's own SQLSTATE before the exception type
is consulted, because the two cases that matter most are indistinguishable otherwise.
A transaction the server rolled back for contention — `40001` when it could not
serialise, `40P01` on a deadlock, `55P03` on a lock timeout — arrives as an
`OperationalError` exactly like a dropped connection does. PostgreSQL documents retrying
these as the remedy rather than a workaround, so they are retried; reading only the
exception type treated them as a statement the server refused and gave up on the first
attempt. They are reported as a server failure because the server is what aborted the
work, and a kind of their own would have to be named by every domain to say the same
thing.

`OperationalError` is otherwise left out of the map deliberately: it covers both a
dropped connection and a statement the server refused, so the pool's own verdict
decides, and anything it did not invalidate is treated as permanent rather than retried
into the same wall. A cancelled statement — `57014`, which a `statement_timeout` raises
— stays on that path on purpose. A timeout does not say whether the server finished the
work, so repeating it risks applying it twice.

`SQLService` reaches the same database through the same reader, and is the one backend
where the retry had to be split rather than applied whole. Every method opens its own
session from `sessionmaker.begin()`, which commits on a clean exit and rolls back on an
exception, so one method is one complete transaction and a retry enters a new one. That
matters because a failed session cannot be resumed — SQLAlchemy refuses to continue on
one until it is explicitly rolled back, and its own FAQ names retrying the whole
transaction as the approach to take.

Writing through the ORM survives a retry because the unit of work already distinguishes
the two cases this layer could not. A flush that assigned a primary key and then rolled
back leaves the object transient, so the retry inserts; a commit that landed but whose
acknowledgement was lost leaves it detached with its identity, so the retry updates in
place rather than inserting a second row. Both were measured rather than assumed.

What could not be settled that way is a statement the caller wrote. `execute`,
`update_many`, `delete_many` and `execute_in_transaction` carry SQL this layer never
saw, and `DELETE ... WHERE id = :id` repeating is harmless where `UPDATE t SET n = n + 1`
is not. They are wrapped by `classifying` rather than `retrying`: failures still become
the SQL family, so a caller catches one thing whichever method it used, but nothing is
repeated unless the caller passes `retry=True`. `transaction()` and `atransaction()` go
further and are never repeated at all — they hand the caller a session to drive, so the
statements inside are not ours to replay — but they still report their failures as the
same family, which is what lets a store's duplicate check work through them.

That family is also what a store now catches. `AbstractStore._duplicate_error_types()`
exists so a store can tell a collision from a genuine failure, and it previously named
`IntegrityError` directly — which stops arriving the moment the service wraps its
errors, silently, since a missing match is not an error. It names the domain's conflict
class instead, which is what the hook was for: detecting a duplicate without knowing
which client raised it.

One risk retry cannot remove: if a commit lands and the acknowledgement is lost, no
client can tell. A replayed insert against a server-generated key is the case that
quietly duplicates, so a write path that matters should carry a key the client
generated and a unique constraint to go with it, and fail loudly instead.

The document stores draw the same line, for the same reason. `update_one`,
`update_many` and an aggregation take an operation the caller wrote, and `{"$inc": …}`
is MongoDB's version of adding one to a column: repeating it adds two. Those are
classified rather than retried, while reads, a replace, a delete by filter and an insert
keep their retry, because repeating those reaches the same state. A Cosmos aggregation
runs read-only Cosmos SQL, so it stays a read.

MongoDB is also the one client whose own retry is kept rather than turned off, which is
the opposite of what the model and search clients do. It sends a transaction number the
server uses to recognise a repeat, so a single-document write that lands but is never
acknowledged is applied once rather than twice — the one thing a retry above the driver
cannot do. It covers single-document writes only, which is exactly why `update_many` and
the operations carrying a caller's update document are not repeated at all. Reads are
left to the handler, so the budget, the backoff and the logging stay in one place rather
than multiplying against the driver's own.

Depth is how specific a module is. The package root holds the abstraction and nothing
else: `openai.py` beside it reads the OpenAI SDK but decides nothing about what callers
see. The subfolders bind a domain: `llm/openai.py` and `llm/anthropic.py` raise the LLM
family, `embedding/openai.py` raises the embedding family. Because the kind an exception
represents is a property of the SDK rather than of the caller, it is stated once and two
domains reuse it.

That vocabulary is `FailureKind`, and it lives in `models/definitions/` with every other
enum rather than inside the handler package — it describes a failure, and the retry loop
is only one of the things that reads it. `RETRYABLE_FAILURES` sits beside it and is what
makes the decision reviewable in one place: throttling, a server fault, a timeout and a
connection failure are properties of the
moment, so the same request a little later plausibly succeeds. A malformed request, a
rejected credential, a missing resource and a conflict are properties of the request,
so a second attempt buys latency and arrives at the identical error. An unrecognised
failure is not retried either, on the grounds that something the SDK never promised is
a poor thing to hammer. Where a provider says how long to wait, that is honoured ahead
of the backoff.

The same arrangement reaches the remaining backends. Cosmos DB reports failures as
`CosmosHttpResponseError`, which derives from the same `HttpResponseError` as object
storage, so `nosql/cosmos.py` and `keyvault/azure.py` are kind-to-class maps over the
existing Azure reader and read no SDK themselves. MongoDB and Redis needed their own,
and both carry an ordering hazard worth naming: Redis derives `AuthenticationError`
from `ConnectionError`, so a map resolved in the wrong order would retry a rejected
password until the budget ran out, while PyMongo reports a rejected credential as an
`OperationFailure` like any other, separated only by its code. A stepping-down primary
is retried deliberately, because finding the new one is how the driver recovers.

A cache is the one backend given a shorter budget than the default. It sits on the
request path and is an optimisation, so waiting out a long backoff costs more than the
miss it was there to avoid.

Every billable call is recorded, and recording sits in the abstraction rather than
in each adapter. It has to happen once per call while an adapter retries inside
itself, so doing it above the retry is the only place it can go — and it means a new
provider cannot forget. That is why `_embed` and `_rerank` return usage alongside
their payload instead of just the payload.

The usage ledger and the debug trace split along what they cost to keep. Records are
small, always on, and safe to aggregate or persist. The trace holds the verbatim
prompt and response, which are large and often sensitive, so it is captured only when
a request asks for it — `?debug=true` on the chat endpoint is what starts it, and an
ordinary request pays nothing. Neither repeats the other: accounting lives on the
record, payloads on the call, and a shared `call_id` joins them.

One record covers one call however many tries it took, because a retried attempt
is not billed and counting it as its own call would overstate how many were made.
What it took rides on the record instead: `attempts`, and one entry per failed
attempt saying what went wrong. A call that was throttled twice and then succeeded
is otherwise indistinguishable from a clean one, which is the case worth seeing —
the failure that ends a call was always recorded, the ones it recovered from were
not. The handler collects that in a `ContextVar`, because one handler serves many
concurrent calls and each task needs its own. It sits in `tools/attempt_trail.py`
alongside `tools/correlation.py`, which is the same shape for the same reason: state
scoped to one unit of work, kept with its accessors rather than inside whichever
service happens to write to it.

Reranking is billed per search unit rather than per token, so its records usually
carry no cost. `UsageRecord.cost_usd` stays `None` when no rate is known, which is
the point: an unpriced call stays visible rather than looking free.

Object storage is deliberately not in the usage ledger. It is not billed per call in
the way that ledger models, and the call volume is an order of magnitude higher, so
it belongs in logs and metrics rather than drowning a per-request trace.

A service whose every method is its own round trip — a blob store has around
twenty — would bury what each method does under what the retry policy does if each
body were wrapped by hand. `retrying` and `aretrying` run a method through the
handler on its `_error_handler`, so the method keeps its shape and the policy stays
in one place. Neither may decorate a generator: calling one returns an iterator
without doing any work, so the failure would happen later, outside the retry loop.
`AzureBlobService.list_blobs` and `RedisCacheService.subscribe` are left undecorated
for exactly that reason. A health check is left alone too: it reports what is true
now, and retrying one would report a failing dependency as healthy for as long as the
backoff lasted.

A decorated method calling another one is common — pagination counts then fetches, a
secret store reads several names — and the handler lets an error that is already a
`ProviderError` through untouched rather than treating it as something unexpected.
Without that, the outer call would re-wrap what the inner call had already worked out,
so the same dropped connection arrived as a connection failure from `query` and an
unexpected failure from `paginate`. The inner call has also already spent its retries,
so there is nothing left for the outer one to repeat.

Nothing computed belongs behind the policy at all. `format_secret_name` applies this
starter's tag prefixes and reaches no vault, so retrying it repeated no call, and
routing it through the handler turned a deliberate `ValueError` about a bad prefix into
a vault failure. Being able to decorate a method is not a reason to.

Those domains are deliberately separate families. `EmbeddingRateLimitError` is not an
`LLMError`, so a service that both embeds and generates can catch a failure of one
without catching the other. Both derive from `ProviderError`, which carries the shape —
status code, request id, the originating exception, and the attempt count when retries
ran out — so a caller that genuinely wants either can still catch one thing.

Declaring a tool sits below the adapters, which render a `Tool` into their provider's shape;
running a turn sits above them, because it needs a model to call. That is why
`base.services.tool_calling` exports `Tool` and `Toolbox` while `ToolRunner` is imported from
`base.services.tool_calling.runner` — re-exporting it would pull the upper layer into every
import of the lower one and close a cycle.
