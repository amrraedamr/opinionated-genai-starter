# 04 — Configuration

> **Requirement:** each app/service has its own `.env` — different deployments, different
> LLM keys. Settings should be a base class that each service inherits and extends, with
> only genuinely shared settings living in `base`.

That is the right instinct. This document refines it in two ways: **compose mixins rather
than inherit one class**, and **resolve settings through a provider** so that `base` code
sees the service's settings rather than its own.

---

## Problem with the current shape

`base/services/configuration/settings.py` is a single 231-line `Settings` class holding
every field for every integration — app, auth, five OAuth providers, enterprise IdP, SQL,
NoSQL, vectors, blob, Key Vault, LLM, cache.

It is exposed as a module-level singleton and imported directly in **20 places**:

```python
from base.services.configuration import settings
```

Two consequences for multiple services:

1. **Inheriting one god-class gives every service every field.** A worker with no HTTP
   surface still carries `oauth_apple_private_key_path`. Worse, you lose the ability to
   *require* a field: `OPENAI_API_KEY` cannot be mandatory in `chat_api` if the same class
   must also validate for a service that has no LLM.
2. **A subclass is invisible to `base`.** If `chat_api` defines
   `class Settings(BaseSettings)`, the 20 modules inside `base` still import the *base*
   instance. The subclass is never consulted.

---

## Design

### Layer 1 — `CoreSettings`: genuinely universal only

Everything here must apply to *every* deployable, including a worker with no database and
no HTTP server. That is about six fields.

```python
# packages/base/base/services/configuration/core.py
from pydantic_settings import BaseSettings, SettingsConfigDict

from base.services.common.meta import SingletonSettings


class CoreSettings(BaseSettings, metaclass=SingletonSettings):
    """Settings every deployable has, regardless of shape."""

    model_config = SettingsConfigDict(
        env_file=(".env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str                       # required — identifies the service everywhere
    environment: str = "local"
    log_level: str = "INFO"
```

`service_name` is deliberately **required and unnamed-by-default**. It is the namespace key
for Redis channels (`CacheChannels.for_app`), cache keys, queue names, log fields and
trace attributes — a wrong default here causes cross-service collisions, so the service
must state it.

> **Nothing subsystem-specific belongs here.** Serving knobs (`api_prefix`, `api_host`,
> `api_port`, `max_gunicorn_workers`) live in `ServerSettingsMixin`, because they pair with
> the `server` extra — a broker-consuming worker installs neither uvicorn nor gunicorn and
> must not carry a worker count it can never act on. The test for `CoreSettings` membership
> is: *would a worker with no HTTP surface and no database still need this?*

### Layer 2 — mixins per subsystem

One mixin per optional subsystem, mirroring the dependency extras in
[doc 02](02-library-boundary.md). A service includes a mixin only if it uses that subsystem.

```python
# packages/base/base/services/configuration/mixins.py
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class ServerSettingsMixin(BaseModel):
    """HTTP serving. Pairs with the `server` extra (uvicorn + gunicorn)."""

    api_prefix: str = "/api"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_gunicorn_workers: int = 2


class SQLSettingsMixin(BaseModel):
    sql_backend: Literal["sqlite", "mysql", "postgresql"] = "sqlite"
    sqlite_db_path: str = "database/app.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    db_ssl_enabled: bool = True


class PostgresSettingsMixin(BaseModel):
    pg_host: str = ""
    pg_port: str = "5432"
    pg_database: str = ""
    pg_ssl_root_cert: str = ""
    pg_username: str = Field(
        default="", validation_alias=AliasChoices("POSTGRES_USER", "PG_USERNAME")
    )
    pg_password: str = Field(
        default="", validation_alias=AliasChoices("POSTGRES_PASSWORD", "PG_PASSWORD")
    )


class TokenVerificationMixin(BaseModel):
    """Verify tokens. Does NOT issue them — see doc 05."""
    auth_public_key_path: str = "keys/public.pem"


class TokenIssuingMixin(TokenVerificationMixin):
    """Only a service that issues tokens should include this."""
    auth_private_key_path: str = "keys/private.pem"
    access_token_expiry_seconds: int = 900
    refresh_token_expiry_seconds: int = 604800
```

The full set, each lifting one group from today's file **verbatim**:
`SQLSettingsMixin`, `PostgresSettingsMixin`, `MySQLSettingsMixin`, `RedisSettingsMixin`,
`LLMSettingsMixin`, `EmbeddingSettingsMixin`, `TokenVerificationMixin`,
`TokenIssuingMixin`, `OAuthSettingsMixin`, `IdpResourceServerMixin`,
`MongoSettingsMixin`, `CosmosSettingsMixin`, `BlobSettingsMixin`,
`KeyVaultSettingsMixin`, `VectorSettingsMixin`, `AzureSearchSettingsMixin`,
`PromptSettingsMixin`.

> **Field names are unchanged, deliberately.** Earlier drafts of this document sketched
> generic names (`db_host`, `db_name`, `openai_api_key`, `default_model`). Renaming would
> break every existing `.env`, every deployment variable, and the `AliasChoices` that let
> `POSTGRES_USER` and `PG_USERNAME` both work — for no architectural gain. The grouping is
> the point; the names are not. `pg_*` and `mysql_*` are also what makes
> `PostgresSettingsMixin` and `MySQLSettingsMixin` separable at all.
>
> The one rename that *was* worth it: `app_name` becomes `service_name`, because it is now
> the cross-service namespace key and must be stated rather than defaulted.

### Layer 3 — the service composes and extends

```python
# services/chat_api/chat_api/settings.py
from pydantic_settings import SettingsConfigDict

from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    IdpResourceServerMixin,
    LLMSettingsMixin,
    OAuthSettingsMixin,
    PostgresSettingsMixin,
    PromptSettingsMixin,
    RedisSettingsMixin,
    SQLSettingsMixin,
    ServerSettingsMixin,
    TokenIssuingMixin,
)


class ChatApiSettings(
    CoreSettings,
    ServerSettingsMixin,
    SQLSettingsMixin,
    PostgresSettingsMixin,
    RedisSettingsMixin,
    LLMSettingsMixin,
    # chat_api still issues tokens and owns users. Phase 5 moves that to the
    # identity service, at which point this drops to TokenVerificationMixin.
    TokenIssuingMixin,
    OAuthSettingsMixin,
    IdpResourceServerMixin,
    PromptSettingsMixin,
):
    model_config = SettingsConfigDict(
        env_file=("../../.env.shared", ".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "chat_api"
    api_prefix: str = "/api/7r"

    cors_origins: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = ChatApiSettings()
```

Verified: `ChatApiSettings` carries `sql_backend`, `pg_database`, `redis_url`,
`openai_llm_api_key`, `auth_private_key_path` and `oauth_providers`, and carries **no**
`qdrant_url`, `mongodb_host`, `azure_cosmos_account_uri`, `azure_search_endpoint`,
`mysql_host`, `azure_blob_account_url`, `keyvault_url` or `openai_embedding_model`.

> **On tightening a field to required.** An earlier draft made `openai_api_key`
> `Field(min_length=1)` here. That is the right move for a real deployment and the wrong
> one for a starter: it would stop `chat_api` booting on a fresh clone before any key
> exists, and the whole regression suite — register, login, conversations — needs no LLM
> call. The mechanism is what matters and it is available; the starter leaves the field
> optional and documents tightening it as a per-deployment decision.

This is the inheritance model asked for — just granular enough that each service carries
only what it uses, and can **tighten** validation (`openai_api_key` is mandatory here and
does not exist at all in `identity`).

---

## Making `base` see the service's settings

`base` modules cannot import a service's settings — that would invert the dependency. Two
mechanisms, used together.

### Mechanism 1 — provider registration (runtime)

The existing `SingletonSettings` metaclass and `Singleton.replace()` already provide the
hook.

```python
# packages/base/base/services/configuration/__init__.py
_settings: CoreSettings | None = None


def configure(instance: CoreSettings) -> CoreSettings:
    """Called once by the service composition root, before anything else."""
    global _settings
    _settings = instance
    return instance


def get_settings() -> CoreSettings:
    if _settings is None:
        raise RuntimeError(
            "Settings not configured. Call "
            "base.services.configuration.configure(YourSettings()) in your "
            "service entrypoint before using any base service."
        )
    return _settings


def is_configured() -> bool: ...
def reset() -> None: ...        # for tests
```

Every `settings.foo` site in `base` becomes `get_settings().foo`. **No deprecation shim**
was needed: the old module-level `settings` singleton is deleted outright, because every
call site reads its fields inside a function body rather than at import time, so the
migration is total and the failure mode for a missed one is an immediate `ImportError`
rather than a silently stale object.

The composition root registers settings before importing anything that reads them:

```python
# services/chat_api/chat_api/app.py
from base.services.configuration import configure

from chat_api.settings import settings

configure(settings)

from base.app import create_app  # noqa: E402 — must follow configure()
...
```

`alembic/env.py` hands its settings to `base.tools.migrations.run_migrations`, which calls
`configure()` the same way, so migrations and the running service resolve configuration
identically.

### Mechanism 2 — Protocols (static)

Runtime lookup alone loses type safety. Each `base` subsystem declares the shape it needs,
and type checkers verify structurally that a service's composed settings satisfy it:

```python
# packages/base/base/services/sql/protocol.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsSQLConfig(Protocol):
    sql_backend: str
    sqlite_db_path: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_timeout: int
    db_pool_recycle: int
    db_ssl_enabled: bool
```

```python
class SQLService:
    def __init__(self, config: SupportsSQLConfig | None = None) -> None:
        self._config = config or cast(SupportsSQLConfig, get_settings())
        self._backend = SQLBackend(self._config.sql_backend)
        self._client = SQLClient(self._backend, self._config, _key=self._backend.value)
```

Shipped: `SupportsSQLConfig` / `SupportsPostgresConfig` / `SupportsMySQLConfig`,
`SupportsLLMConfig` / `SupportsEmbeddingConfig`, `SupportsCacheConfig`,
`SupportsVectorConfig` / `SupportsAzureSearchConfig`, and `SupportsServerConfig`. They are
`runtime_checkable`, so the conformance is also assertable in a test — `ChatApiSettings`
satisfies the SQL, Postgres, LLM, cache and server protocols and does **not** satisfy the
vector one.

`SupportsServerConfig` earns its place: the gunicorn worker count is read only on the
non-`local` branch of `run_webserver`, so a service that forgot `ServerSettingsMixin` would
pass every local test and then fail on container start. The protocol moves that to
type-check time.

> **`_key` matters here.** `SQLClient` is a `KeyedSingleton` whose default identity key is
> `hash_args` over its constructor arguments. Passing a whole settings object would
> serialize every field on every lookup, so the backend name is passed as an explicit
> `_key` instead.

A service that uses `SQLService` without `SQLSettingsMixin` now fails at type-check time
with a clear message, rather than at runtime with `AttributeError`.

> **Preferred for new code:** pass config explicitly into constructors. The provider is the
> pragmatic bridge for the 20 existing call sites, not the long-term ideal.

---

## Per-service `.env`

### Layout

```
.env.shared.example          # local-dev infra shared by all services (committed)
.env.shared                  # gitignored
services/identity/.env.example
services/identity/.env       # gitignored
services/chat_api/.env.example
services/chat_api/.env       # gitignored
```

The root `.gitignore` applies at every depth, but it needed one fix: `.env.*` swallows
`.env.shared.example`, because `!.env.example` un-ignores only that exact name. Without the
second negation the committed shared template is silently never added.

```
.env
.env.*
!.env.example
!.env.*.example
```

### Precedence

`pydantic-settings` applies `env_file` entries in order, with **later files winning**, and
real process environment variables beating all files.

| Priority | Source | Purpose |
| --- | --- | --- |
| 1 (highest) | Process environment | Production. Injected by Container Apps / K8s / App Service |
| 2 | `.env.local` | Personal overrides. Never committed |
| 3 | `.env` | The service's own local values — **its own LLM key, its own database** |
| 4 | `.env.shared` | Shared local infra only (Postgres host, Redis URL) |
| 5 (lowest) | Field defaults | Safe defaults in code |

`.env.shared` is strictly a **local developer convenience** for infrastructure endpoints. It
must never hold a secret that differs per environment — in deployment it does not exist,
and process env supplies everything.

### Different keys per service, per deployment

This falls out naturally. `chat_api` and `ingestion-worker` each read `OPENAI_API_KEY` from
their **own** `.env`, so they can use different keys, different quotas, different models:

```ini
# services/chat_api/.env
SERVICE_NAME=chat_api
OPENAI_API_KEY=sk-chat-...
DEFAULT_MODEL=gpt-4o
DB_NAME=chat_db
```

```ini
# services/ingestion-worker/.env
SERVICE_NAME=ingestion-worker
OPENAI_API_KEY=sk-batch-...
DEFAULT_MODEL=gpt-4o-mini
MONGO_DATABASE=ingestion
```

In production each service is a separate deployment unit with its own environment
configuration, so the same separation holds with no extra mechanism.

### Compose: keep the files, don't flatten them

```yaml
chat_api:
  build: { context: ., dockerfile: docker/service.Dockerfile, args: { SERVICE: chat_api } }
  env_file:
    - ./.env.shared
    - ./services/chat_api/.env
  environment:
    ENVIRONMENT: local
```

Compose's `env_file` honours the same ordering, so one `docker compose up` reproduces the
local precedence exactly.

> **If you ever must put several services in one env file**, use `env_prefix` per settings
> class (`CHAT_API_`, `IDENTITY_`) to disambiguate. Prefer separate files; the prefix is a
> fallback, not the default.

---

## Secrets

| Environment | Mechanism |
| --- | --- |
| Local | `.env` / `.env.local`, gitignored |
| CI | Repository/environment secrets as process env |
| Production | Key Vault, K8s Secrets, or Container Apps secrets → injected as process env |

`.env` files never ship inside an image. `.dockerignore` must exclude them, and
[doc 06](06-build-and-deploy.md) treats that as a build-time assertion rather than a
convention.

The existing `Settings` docstring already sketches a Key Vault settings source plus
`CacheChannels.KEYVAULT_RELOAD` pub/sub reload. That design carries over unchanged — add it
as a source on `CoreSettings` so **every** service inherits hot reload at once, and
namespace the reload channel by `service_name` so one service's reload does not wake the
rest.

---

## Validation at startup

Fail fast and loudly. A service that boots with a missing key and only fails on the first
user request is worse than one that refuses to start.

- Mark truly required fields with no default (`service_name`, and per-service keys like
  `openai_api_key` above).
- Instantiate settings in the composition root **before** `create_app()` returns.
- Surface Pydantic's validation error directly — it already names the offending field.
- Add a `uv run --package chat_api python -m chat_api.settings --check` entrypoint for CI
  and for a container `preStart` probe.
