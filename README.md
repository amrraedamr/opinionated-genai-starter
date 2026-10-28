# Opinionated GenAI Starter

A production-ready FastAPI starter with a layered architecture, metaclass-based dependency injection, multi-database support, and pre-wired integrations for Azure services, Redis, vector databases, and LLMs.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Configuration](#configuration)
- [Dependency Injection (Metaclass System)](#dependency-injection-metaclass-system)
- [Services](#services)
  - [SQL Databases](#sql-databases)
  - [NoSQL (MongoDB, CosmosDB)](#nosql-mongodb-cosmosdb)
  - [Redis Cache](#redis-cache)
  - [Blob Storage](#blob-storage)
  - [Key Vault](#key-vault)
  - [Vector Search](#vector-search)
  - [Text Embeddings](#text-embeddings)
  - [LLM (OpenAI)](#llm-openai)
  - [Prompt Management](#prompt-management)
  - [Chat History](#chat-history)
  - [Conversations](#conversations)
  - [Background Tasks](#background-tasks)
  - [HTTP Client](#http-client)
  - [Logging](#logging)
- [Agents (AbstractAgent)](#agents-abstractagent)
- [Execution Context](#execution-context)
- [Debug Trace](#debug-trace)
- [Localization & Canned Messages](#localization--canned-messages)
- [Error Handling](#error-handling)
- [Authentication](#authentication)
- [Store (Repository) Layer](#store-repository-layer)
- [Models](#models)
  - [ORM Models](#orm-models)
  - [View Models (Pydantic)](#view-models-pydantic)
  - [Definitions (Enums, Constants)](#definitions-enums-constants)
- [Routers](#routers)
- [Tools (Utilities)](#tools-utilities)
- [Streaming (SSE)](#streaming-sse)
- [Live Log Streaming](#live-log-streaming)
- [Webserver](#webserver)
- [API Endpoints](#api-endpoints)
- [Tech Stack](#tech-stack)
- [Multi-Service Architecture (design)](#multi-service-architecture-design)

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- RSA key pair for JWT (see below)

### Installation

```bash
uv sync
```

### Generate RSA Keys (JWT Auth)

```bash
mkdir -p keys
openssl genrsa -out keys/private.pem 2048
openssl rsa -in keys/private.pem -pubout -out keys/public.pem
```

### Configure

1. Copy `.env.example` → `.env` and fill in what you need. Typed defaults live in `Settings` (`base/services/configuration/settings.py`) — you only set the integrations you use.
2. **SQLite (default):**
   - `mkdir -p database`
   - keep `SQL_BACKEND=sqlite` and `SQLITE_DB_PATH=database/app.db` (already in `.env.example`)
   - Alembic creates the `.db` file on first `alembic upgrade head`
3. In Docker/deploy, set the same variables as real environment vars instead of a `.env` file.

### Run Migrations

```bash
# Apply all migrations (creates/updates database schema)
uv run alembic upgrade head
```

### Run

```bash
# Using the uv-managed environment
uv run api

# Or directly with uvicorn
uv run uvicorn base.main:app --reload
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### Database Migrations (Alembic)

Schema changes are managed via Alembic. The migration environment reads `SQL_BACKEND` from Settings (env / `.env`) to target the correct database.

```bash
# Apply all pending migrations
uv run alembic upgrade head

# Generate a new migration after changing ORM models
uv run alembic revision --autogenerate -m "description of change"

# Rollback one migration
uv run alembic downgrade -1

# View current migration status
uv run alembic current
```

---

## Project Structure

```
.
├── base/
│   ├── __init__.py                       # Entry point (calls run_webserver)
│   ├── main.py                           # FastAPI app, lifespan, register_request middleware, routes
│   │
│   ├── clients/                          # Low-level database/service clients
│   │   ├── blob/
│   │   │   └── azure_blob.py             # Azure Blob Storage client (sync + async)
│   │   ├── nosql/
│   │   │   ├── cosmosdb.py               # Azure Cosmos DB client (SQL API, sync + async)
│   │   │   └── mongodb.py                # MongoDB client (PyMongo sync + async)
│   │   ├── redis/
│   │   │   └── __init__.py               # Redis client (aioredis async)
│   │   ├── sql/
│   │   │   └── client.py                 # SQLClient (unified, driven by SQLBackend enum)
│   │   └── vectors/
│   │       ├── azure_ai_search.py        # Azure AI Search client
│   │       └── qdrant.py                 # Qdrant vector DB client
│   │
│   ├── models/
│   │   ├── definitions/                  # Enums, constants, error definitions
│   │   │   ├── background_tasks.py       # TaskStatus, BackgroundTaskBackend
│   │   │   ├── cache.py                  # Cache channel enums
│   │   │   ├── canned_messages.py        # CannedMessages — bilingual user-facing strings
│   │   │   ├── database.py               # SQLBackend, DatabaseTable, Locale, SortingOrder
│   │   │   ├── errors.py                 # Auth, Database, Generic error enums
│   │   │   ├── exceptions.py             # Exception categories
│   │   │   ├── message.py                # MessageRole (system/developer/user/assistant/tool)
│   │   │   ├── model.py                  # LLM provider type enum
│   │   │   └── vector_search.py          # Vector search distance metrics
│   │   ├── orm/                          # SQLAlchemy ORM models
│   │   │   ├── __init__.py               # Base, TimestampMixin, auto-import
│   │   │   ├── chat_history_message.py   # ChatHistoryMessageDTO model
│   │   │   ├── conversation.py           # ConversationDTO model
│   │   │   ├── prompt.py                 # PromptDTO, ActivePromptDTO models
│   │   │   ├── user.py                   # UserDTO model (password optional for social)
│   │   │   └── user_identity.py          # UserIdentityDTO — (provider, subject) → user
│   │   └── view_models/                  # Pydantic request/response models
│   │       ├── __init__.py               # BaseModel, BaseResponse, ErrorResponse, DebugContext
│   │       ├── auth.py                   # Auth + OAuth request/response models, token types
│   │       ├── background_tasks.py       # TaskHandle
│   │       ├── chat.py                   # SendChatRequest/Response, ChatMessageView, ListChatResponse
│   │       ├── chat_history.py           # StoredMessage, StoredToolCall, ConversationId, ChatHistoryConfig
│   │       ├── conversation.py           # CRUD request/response models for conversations
│   │       ├── database.py               # Pagination, sorting, filtered params
│   │       ├── debug_trace.py            # DebugCall, FlowEvent, DebugTrace
│   │       ├── execution_context.py      # FlowVariables
│   │       ├── flow.py                   # FlowAbortError
│   │       ├── http.py                   # HttpResponse model
│   │       ├── llm_exception.py          # LLMError hierarchy
│   │       ├── llm_result.py             # CompletionResult, ResponseResult, Parsed*Result
│   │       ├── model.py                  # ModelConfigurations, EmbeddingConfigurations
│   │       ├── models.py                 # OpenAIModels enum
│   │       ├── prompt/                   # Prompt subpackage
│   │       │   ├── __init__.py           # Re-exports
│   │       │   ├── prompt.py             # Prompt view model
│   │       │   ├── prompt_template.py    # PromptTemplate, *MessageTemplate
│   │       │   └── rendered_prompt.py    # RenderedPrompt, RenderedPromptInstruction/Message
│   │       ├── streams.py                # SSE stream message models
│   │       ├── user.py                   # User CRUD request/response models
│   │       └── vector_search.py          # Vector search result models
│   │
│   ├── routers/                          # FastAPI routers (auto-discovered)
│   │   ├── auth.py                       # /auth — register, login, refresh, oauth
│   │   ├── conversation.py               # /conversations — CRUD + chat send/list
│   │   └── user.py                       # /users — get/edit/delete me
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── __init__.py               # AbstractAgent template-method base
│   │   │   └── test.py                   # TestAgent example
│   │   ├── auth/
│   │   │   ├── __init__.py               # AuthService — JWT + optional IdP Bearer
│   │   │   ├── auths.py                  # HTTPBearer scheme
│   │   │   ├── dependencies.py           # avalidate_user / validate_user_dependency
│   │   │   └── oauth/                    # Login-with-X (OIDC/OAuth providers)
│   │   │       ├── __init__.py           # OAuthService
│   │   │       └── providers.py          # Provider registry + profile fetch
│   │   ├── background_tasks/
│   │   │   ├── __init__.py               # AbstractBackgroundTaskService, RegisteredTask
│   │   │   └── in_memory.py              # InMemoryBackgroundTaskService
│   │   ├── blob/
│   │   │   ├── __init__.py               # AbstractBlobService
│   │   │   ├── azure_blob.py             # AzureBlobService implementation
│   │   │   └── memory.py                 # InMemoryBlobService (for testing)
│   │   ├── cache/
│   │   │   ├── __init__.py               # AbstractCacheService
│   │   │   ├── memory.py                 # InMemoryCacheService (for testing)
│   │   │   └── redis.py                  # RedisCacheService implementation
│   │   ├── chat_history/
│   │   │   └── __init__.py               # ChatHistoryService (KeyedSingleton)
│   │   ├── common/
│   │   │   └── meta/
│   │   │       └── __init__.py           # Metaclass DI system
│   │   ├── configuration/
│   │   │   ├── __init__.py               # exports Settings, settings
│   │   │   └── settings.py               # Settings (BaseSettings + SingletonSettings)
│   │   ├── execution_context/
│   │   │   └── __init__.py               # ExecutionContext (Stored — per-request scope)
│   │   ├── http/
│   │   │   └── __init__.py               # HttpRequestService (httpx + requests)
│   │   ├── keyvault/
│   │   │   ├── __init__.py
│   │   │   └── azure_keyvault.py         # AzureKeyVaultService (concrete singleton)
│   │   ├── llm/
│   │   │   ├── __init__.py               # OpenAIService, OpenAIMessageAdapter re-exports
│   │   │   ├── openai_service.py         # OpenAIService (Chat Completions + Responses, debug capture)
│   │   │   ├── openai_adapter.py         # OpenAIMessageAdapter (StoredMessage/RenderedPrompt → OpenAI)
│   │   │   └── error_handler/
│   │   │       ├── __init__.py
│   │   │       └── openai_handler.py     # OpenAIErrorHandler (retry + classify + content filter)
│   │   ├── logging/
│   │   │   ├── __init__.py               # LogMessage type alias
│   │   │   ├── config.py                 # Colored logging config
│   │   │   ├── console.py                # ConsoleLoggerService (concrete singleton)
│   │   │   ├── helper/
│   │   │   │   └── log_builder.py        # Log formatting helper
│   │   │   └── tee_stream/
│   │   │       └── __init__.py           # TeeStream + TerminalCaptureService
│   │   ├── nosql/
│   │   │   ├── __init__.py               # AbstractNoSQLService (SDK-agnostic)
│   │   │   ├── cosmosdb.py               # CosmosDBService implementation
│   │   │   └── mongodb.py                # MongoDBService (+ Atlas Search vector index helpers)
│   │   ├── prompt/
│   │   │   ├── __init__.py
│   │   │   ├── decorator.py              # @register_prompt decorator
│   │   │   ├── loader.py                 # PromptLoader (singleton, pub/sub reload)
│   │   │   ├── service.py                # PromptService — get/format/render
│   │   │   └── templates/                # Prompt template definitions
│   │   ├── sql/
│   │   │   ├── __init__.py               # AbstractSQLService (query, paginate, upsert, transactions)
│   │   │   └── service.py                # SQLService (unified, driven by SQLBackend enum)
│   │   ├── store/
│   │   │   ├── __init__.py               # AbstractStore + AbstractSequenceStore
│   │   │   ├── chat_history/
│   │   │   │   ├── __init__.py           # ChatHistoryStore marker
│   │   │   │   ├── memory.py             # InMemoryChatHistoryStore (for tests)
│   │   │   │   └── sql.py                # SqlChatHistoryStore
│   │   │   ├── conversation.py           # ConversationStore (with *_for_user variants)
│   │   │   ├── user.py                   # UserStore implementation
│   │   │   └── user_identity.py          # UserIdentityStore (provider + subject lookup)
│   │   ├── text_embedding/
│   │   │   ├── __init__.py
│   │   │   └── openai_embedding.py       # OpenAITextEmbeddingService (concrete keyed singleton)
│   │   └── vector_search/
│   │       ├── __init__.py               # AbstractVectorSearchService
│   │       ├── azure_ai_search.py        # AzureAISearchService implementation
│   │       ├── mongodb.py                # MongoDBVectorSearchService (Atlas $vectorSearch)
│   │       └── qdrant.py                 # QdrantService implementation
│   │
│   └── tools/                            # Utility modules
│       ├── __init__.py
│       ├── database.py                   # SQLResultFormats (query result processors)
│       ├── deserializers.py              # JSON deserialization helpers
│       ├── errors.py                     # ErrorsException, raise_error
│       ├── exception_handlers.py         # AppExceptionHandler (FastAPI error mapping)
│       ├── hashing.py                    # hash_args (deterministic SHA-256 over arbitrary args)
│       ├── locale.py                     # get_text_locale (EN/AR detection)
│       ├── misc.py                       # JSON serialization, general helpers
│       ├── prompt.py                     # extract_template_variables, safe_format
│       ├── routers.py                    # Auto-discovery router registration
│       ├── security.py                   # PasswordHashingService (Argon2)
│       ├── serializers.py                # ExpansiveEncoder, serialize()
│       ├── stream.py                     # StreamIterator (SSE support)
│       ├── tokenizer.py                  # TokenCounter (tiktoken-backed, KeyedSingleton)
│       ├── validators.py                 # Password validation
│       └── webserver.py                  # Uvicorn/Gunicorn launcher
│
├── database/
│   └── app.db                            # SQLite database file
├── keys/
│   ├── private.pem                       # RSA private key (JWT signing)
│   └── public.pem                        # RSA public key (JWT verification)
│
├── alembic/                              # Alembic migrations
│   ├── env.py                            # Migration environment (Settings-driven)
│   ├── script.py.mako                    # Migration file template
│   └── versions/                         # Migration scripts
│
├── alembic.ini                           # Alembic configuration
├── .env                                  # Local config + secrets (gitignored)
├── .env.example                          # Documented env var template
├── GETTING_STARTED.md                    # Friendly intro: what it is + how to run + try it
├── pyproject.toml                        # uv project metadata & dependencies
└── Dockerfile
```

---

## Architecture Overview

The starter follows a layered architecture with strict separation of concerns:

```
Routers (thin controllers)
   ↓ Depends()
Services (business logic, singletons via metaclass DI)
   ↓
Stores (repository pattern, generic CRUD)
   ↓
SQL/NoSQL Services (session management, query execution)
   ↓
Clients (low-level connection management)
   ↓
Settings (pydantic-settings singleton from env / `.env`)
```

**Key principles:**
- Routes contain zero business logic — they delegate to services
- All services use metaclass-based DI (no manual wiring, no DI container)
- Every database client provides both sync and async interfaces
- Abstract base classes define contracts only where multiple backends exist; single-implementation services are concrete
- Configuration is one typed `settings` object (`BaseSettings` + `SingletonSettings`), loaded from environment variables and optional `.env`. Key Vault is an optional runtime service, not a parallel config bag.

---

## Configuration

Typed settings live in `base/services/configuration/settings.py` as a `BaseSettings` singleton (`SingletonSettings` metaclass). Import the instance:

```python
from base.services.configuration import settings

print(settings.sql_backend)
```

### Sources

| Priority | Source | Notes |
|----------|--------|--------|
| 1 | Process environment | Deploy / Docker / CI |
| 2 | `.env` (optional) | Local only; gitignored. Copy from `.env.example` |

Same `Settings` model everywhere — change `ENVIRONMENT=local|dev|prod|…` (and the rest) in the process env or `.env`.

### Common variables

Full list + comments: [`.env.example`](.env.example). Highlights:

| Variable | Default | Used by |
|----------|---------|---------|
| `ENVIRONMENT` | `local` | Webserver (uvicorn vs gunicorn), error debug detail |
| `APP_NAME` | `opinionated-genai-starter` | Log channel naming |
| `AUTH_PRIVATE_KEY_PATH` / `AUTH_PUBLIC_KEY_PATH` | `keys/*.pem` | `AuthService` JWT |
| `OAUTH_PROVIDERS` + `OAUTH_*_CLIENT_*` | off | Login-with-X (Google/Facebook/Microsoft/GitHub/Apple) |
| `IDP_RESOURCE_SERVER_ENABLED` + `IDP_*` | off | Accept enterprise IdP Bearer JWTs |
| `SQL_BACKEND` | `sqlite` | `SQLService`, Alembic |
| `SQLITE_DB_PATH` | `database/app.db` | SQLite URL |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `PG_*` | — | PostgreSQL |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_*` | — | MySQL |
| `REDIS_URL` | — | `RedisClient` |
| `OPENAI_*` / `AZURE_OPENAI_*` | — | LLM + embeddings |
| `KEYVAULT_URL` / `SYSTEM_KEYVAULT_TAG` / `MODEL_KEYVAULT_TAG` | tags default `appconfig` / `modelconfig` | `AzureKeyVaultService` only |

Webserver bind still uses `API_HOST` / `API_PORT` (read in `run_webserver`, not on `Settings`).

### Key Vault vs Settings

- **Settings** = boot config (env / `.env`).
- **`AzureKeyVaultService`** = runtime vault CRUD for product features.
- Vault-as-a-settings-source (custom `PydanticBaseSettingsSource` + `CacheChannels.KEYVAULT_RELOAD`) is intentionally not built; the recipe is on the `Settings` and `AzureKeyVaultService` docstrings.

---

## Dependency Injection (Metaclass System)

Located in `base/services/common/meta/__init__.py`. All DI is handled through Python metaclasses — no container, no decorators, no manual wiring.

### Metaclass Types

| Metaclass | Behavior | Use Case |
|-----------|----------|----------|
| `Singleton` | One instance per class | Services, clients |
| `SingletonABC` | Singleton + ABC | Abstract service base classes |
| `SingletonSettings` | Singleton + Pydantic ModelMetaclass | Settings models |
| `Stored` | One instance per context scope (via `contextvars`) | Per-request scoped services |
| `StoredABC` | Stored + ABC | Abstract scoped services |
| `KeyedSingleton` | One instance per (class, key) pair; auto-derives key from args when `_key` is not provided | Named instances, config-driven services |
| `KeyedSingletonABC` | KeyedSingleton + ABC | Abstract keyed services |
| `ContextManager` | Auto-injects `__enter__`/`__exit__` | Classes needing context manager protocol |
| `ContextManagerSingleton` | ContextManager + Singleton | Singleton context managers |
| `ContextManagerABC` | ContextManager + ABC | Abstract context managers |
| `ContextManagerStored` | ContextManager + Stored | Scoped context managers |
| `NonInitialize` | Prevents instantiation, returns the class itself | Utility/static-only classes |

### Usage Examples

```python
# Singleton service — always returns same instance
class AuthService(metaclass=Singleton):
    ...

auth = AuthService()  # Always the same instance

# Keyed singleton with explicit key
class TerminalCaptureService(metaclass=KeyedSingleton):
    ...

capture = TerminalCaptureService(_key="MyApp")

# Keyed singleton with auto-derived key — same args = same instance
# Key is generated by serializing args via ExpansiveEncoder + SHA-256 hash
class OpenAIService(metaclass=KeyedSingleton):
    def __init__(self, config: ModelConfigurations) -> None: ...

svc_a = OpenAIService(config=gpt4_config)   # Creates new instance
svc_b = OpenAIService(config=gpt4_config)   # Same config → same instance (cached)
svc_c = OpenAIService(config=mini_config)   # Different config → new instance

# Non-instantiable utility class
class PasswordHashingService(metaclass=NonInitialize):
    @classmethod
    def hash(cls, password: str) -> str: ...

PasswordHashingService.hash("secret")  # Direct class method call

# Scoped per-request
class RequestContext(metaclass=Stored):
    _processor: ClassVar[bool] = False  # Bound to request scope
```

### Global Operations

```python
from base.services.common.meta import purge_all, Singleton, Stored

Singleton.purge()       # Clear all singletons
Stored.delete_scope()   # Clear current scope
purge_all()             # Clear everything
```

---

## Services

### SQL Databases

**Abstract base:** `AbstractSQLService` in `base/services/sql/__init__.py`

Provides a complete SQL toolkit: `query`, `aquery`, `execute`, `aexecute`, `paginate`, `apaginate`, `upsert_row`, `aupsert_row`, `upsert_rows`, `aupsert_rows`, `health_check`, `ahealth_check`, and bulk operations.

**Unified implementation:** A single `SQLClient` (`base/clients/sql/client.py`) and `SQLService` (`base/services/sql/service.py`) handle all three backends, driven by the `SQLBackend` enum. The active backend is `settings.sql_backend`.

| Backend | Sync Driver | Async Driver | Config Keys |
|---------|-------------|--------------|-------------|
| `sqlite` | `pysqlite` | `aiosqlite` | `sqlite_db_path` |
| `mysql` | `pymysql` | `aiomysql` | `mysql_*`, `db_*` |
| `postgresql` | `psycopg2` | `asyncpg` | `pg_*`, `db_*` |

`SQLClient` uses `KeyedSingleton` (keyed by backend type) and manages both sync and async SQLAlchemy engines with session factories, connection pooling, and SSL support. `SQLService` resolves the backend from config automatically — callers just use `SQLService()`. Schema is managed via Alembic migrations — run `alembic upgrade head` to apply.

### NoSQL (MongoDB, CosmosDB)

**Abstract base:** `AbstractNoSQLService` in `base/services/nosql/__init__.py` — SDK-agnostic abstract class with full CRUD, aggregation, pagination, count, distinct, and health check operations. All methods have sync and async variants.

**Implementations:**

| Service | Client | Driver | Config Keys |
|---------|--------|--------|-------------|
| `MongoDBService` | `MongoDBClient` | PyMongo (sync + async) | `mongodb_*` |
| `CosmosDBService` | `CosmosDBClient` | Azure Cosmos DB SDK (SQL API) | `cosmosdb_*` |

Operations: `find_one`, `find_many`, `insert_one`, `insert_many`, `update_one`, `update_many`, `replace_one`, `delete_one`, `delete_many`, `count`, `distinct`, `aggregate`, `paginate`, `health_check`. All with sync + async variants.

### Redis Cache

**Abstract base:** `AbstractCacheService` in `base/services/cache/__init__.py`

**Implementations:**

| Service | Description | Config Keys |
|---------|-------------|-------------|
| `RedisCacheService` | Redis-backed with full pub/sub, distributed locks, connection pooling | `redis_*` |
| `InMemoryCacheService` | Thread-safe in-memory implementation for testing/local dev | None |

Features:
- Key-value get/set with TTL
- Hashset operations
- List operations
- Distributed locks (Redis) / local locks (in-memory)
- Pub/Sub messaging (publish, subscribe, batch publish)

### Blob Storage

**Abstract base:** `AbstractBlobService` in `base/services/blob/__init__.py`

**Implementations:**

| Service | Description |
|---------|-------------|
| `AzureBlobService` | Azure Blob Storage with SAS URL generation |
| `InMemoryBlobService` | In-memory implementation for testing |

Operations: `upload`, `download`, `delete`, `exists`, `get_properties`, `list_blobs`, `generate_url`, container management. All with sync + async variants.

Config keys: `azure_blob_account_url`

### Key Vault

`AzureKeyVaultService` (`base/services/keyvault/azure_keyvault.py`) — runtime vault client (get/set/list) with sync + async SDK wrappers, `DefaultAzureCredential`, and tagged secret naming (`system_keyvault_tag` / `model_keyvault_tag`).

It is **not** a settings backend. Boot config stays on `settings` (env / `.env`). To load Settings from Key Vault later, add a pydantic-settings custom source + optional `CacheChannels.KEYVAULT_RELOAD` pub/sub — see the docstrings on `Settings` and `AzureKeyVaultService`.

Env: `KEYVAULT_URL`, `SYSTEM_KEYVAULT_TAG`, `MODEL_KEYVAULT_TAG`

### Vector Search

**Abstract base:** `AbstractVectorSearchService` in `base/services/vector_search/__init__.py`

**Implementations:**

| Service | Client / Backend | Config Keys |
|---------|------------------|-------------|
| `QdrantService` | `QdrantClient` | `qdrant_*` |
| `AzureAISearchService` | `AzureAISearchClient` | `azure_search_*` |
| `MongoDBVectorSearchService` | `MongoDBClient` (Atlas Search `$vectorSearch`) | `mongodb_*` |

Operations: `search`, `upsert`, `delete`, `get`, `scroll`, `count`, `create_collection`, `delete_collection`, `collection_exists`, `health_check`. All sync + async.

**Atlas Search index management:** `MongoDBService` exposes `create_collection_vector_index()` and `acreate_collection_vector_index()` — idempotent helpers that create or update an Atlas Search vector index for a collection, falling back to drop + recreate (with poll-until-propagated) when an in-place update isn't compatible. Vector size is read from `settings.azure_openai_embedding_dimensions`.

### Text Embeddings

`OpenAITextEmbeddingService` (`base/services/text_embedding/openai_embedding.py`) — concrete keyed singleton that works with any OpenAI-compatible embedding API (OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, Together, Groq, etc.)

Uses `KeyedSingleton` with auto-derived keys: passing the same `EmbeddingConfigurations` returns the cached instance, while a different config creates a new one.

Features:
- Sync (`embed_query`, `embed_documents`) and async (`aembed_query`, `aembed_documents`) variants
- Configurable batch size and optional `dimensions` parameter (for OpenAI `text-embedding-3-*` models)
- HTTP/2 connection pooling via `httpx`
- Azure OpenAI and standard OpenAI client support

Configuration is passed via `EmbeddingConfigurations` Pydantic model (defaults from `settings`; designed so callers can still pass dynamic configs).

### LLM (OpenAI)

`OpenAIService` (`base/services/llm/openai_service.py`) — concrete keyed singleton that works with any OpenAI-compatible API (OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, Together, Groq, etc.).

Uses `KeyedSingleton` with auto-derived keys: passing the same `ModelConfigurations` returns the cached instance, while a different config creates a new one. This means one client + connection pool per unique model configuration, with no manual key management required.

#### API Surfaces

The service supports both OpenAI API surfaces side-by-side, sync and async, streaming and non-streaming:

| Method | Surface | Returns |
|--------|---------|---------|
| `completions` / `acompletions` | Chat Completions | `CompletionResult` |
| `responses` / `aresponses` | Responses API | `ResponseResult` |
| `parse_completions` / `aparse_completions` | Chat Completions (structured) | `ParsedCompletionResult[T]` |
| `parse_responses` / `aparse_responses` | Responses (structured) | `ParsedResponseResult[T]` |

`CompletionResult` and `ResponseResult` expose `.content` (the assistant text) and `.tool_calls` (a normalized `list[StoredToolCall]`) so callers don't have to dig through provider-specific shapes.

#### Streaming

Pass a `StreamIterator` as `stream_iterator=...` to any of the four streaming-capable methods. The service emits `StreamStart`, then `StreamMessage` per delta, then `StreamEnd`, and surfaces `StreamError` on failure (with `error_code`, exception type, and reason).

#### Auto Debug Capture

Every public LLM call records a `DebugCall` (prompt, response content, token counts, duration, error) into the current request's `ExecutionContext.debug_trace` automatically. Callers can pass `label="MyAgent"` or `label="MyAgent:item_id"` so the captured call carries a meaningful identifier for fan-out scenarios. See [Debug Trace](#debug-trace).

#### Error Handling — Domain Hierarchy

Callers never see raw OpenAI SDK exceptions. Everything is wrapped into the `LLMError` hierarchy in `base/models/view_models/llm_exception.py`:

| Class | When |
|-------|------|
| `LLMConfigError` | Missing or invalid configuration (e.g. no `model_name`, no `api_host` for Azure) |
| `LLMBadRequestError` | 400 / 422 |
| `LLMAuthError` | 401 / 403 |
| `LLMNotFoundError` | 404 |
| `LLMConflictError` | 409 |
| `LLMRateLimitError` | 429 after retries exhausted |
| `LLMServerError` | 5xx after retries exhausted |
| `LLMTimeoutError` | Network/API timeout after retries exhausted |
| `LLMConnectionError` | Network connection failure after retries exhausted |
| `LLMContentFilterError` | Pre-flight 400 with `code=="content_filter"` (Azure) or non-streaming response with `incomplete_details.reason=="content_filter"` |
| `LLMIncompleteError` | Response incomplete for any other reason (e.g. `max_tokens`) |
| `LLMStreamError` | Mid-stream failure; `partial_content` may be present |
| `LLMUnexpectedError` | Unanticipated SDK error; check `.provider_error` |

Retry-exhausted variants (`LLMRateLimitError`, `LLMServerError`, `LLMTimeoutError`, `LLMConnectionError`) inherit from `LLMRetryExhausted` and carry an `attempts` count.

#### OpenAIErrorHandler

`base/services/llm/error_handler/openai_handler.py` centralizes classification, retry policy, and domain wrapping behind `.run()` / `.arun()`. Behaviour:

- **Classification:** walks the SDK exception MRO against a table; unknown errors map to `LLMUnexpectedError`. Bad-request errors carrying a `content_filter` code are routed to `LLMContentFilterError`.
- **Retry:** retryable errors (rate limit, 5xx, timeout, connection) get retried up to `retries + 1` attempts.
- **Backoff:** `Retry-After` header on 429 takes precedence; otherwise jittered exponential backoff up to `max_wait`.
- **Final wrap:** retry-exhausted errors are wrapped into the matching `LLMRetryExhausted` subclass with status code, request id, and original provider error preserved.

Per-service tuning is via `ModelConfigurations.api_retries` and `wait_time`.

#### OpenAIMessageAdapter

`base/services/llm/openai_adapter.py` translates the codebase's canonical message types into the shape each API surface needs:

- `to_chat_completions_messages(messages=..., prompt=...)` → `list[ChatCompletionMessageParam]`
- `to_responses_input(messages=..., prompt=...)` → `ResponseInputParam`

Accepts `list[StoredMessage]` from `ChatHistoryService`, a `RenderedPrompt` from `PromptService`, or both. Handles instructions (system / developer), assistant tool calls, and tool-result messages (mapping to `function_call_output` for the Responses API).

**Typical agent pattern.** When the prompt template splices history via a `MessagesPlaceholder`, you pass `prompt=` only — the rendered prompt already contains the prior conversation in the right slot:

```python
rendered = self._prompt_service.render(
    self.PROMPT_KEY,
    history=await self._aget_history(),
    user_query=ctx.flow_variables.user_query,
)
responses_input = OpenAIMessageAdapter.to_responses_input(prompt=rendered)
```

Use the `messages=` argument when you need to append history *after* the prompt's structured messages (e.g. one-shot calls without a chat-aware template), or when you have stored messages but no `RenderedPrompt`.

#### Configuration

Configuration is passed via `ModelConfigurations` (Pydantic; defaults from `settings`). HTTP/2 connection pooling via `httpx`.

### Prompt Management

`PromptService` (`base/services/prompt/service.py`) — concrete singleton backed by `PromptLoader`.

A system for managing LLM prompt templates with decorator-based registration, versioning via database, and distributed hot-reload via Redis pub/sub. The starter ships its own `PromptTemplate` (no LangChain dependency); it accepts both `str.format`-style string templates and structured message lists, validates declared vs detected variables at construction time, and supports `partial_variables`.

**Components:**
- `PromptLoader` — singleton that holds all registered prompts in memory. Auto-discovers and imports all modules under `base/services/prompt/templates/` at startup so `@register_prompt` decorators fire automatically. Uses `InMemoryCacheService` by default (no Redis required); switch to `RedisCacheService` when a Redis instance is available. Subscribes to `PROMPT_RELOAD` pub/sub events for distributed reload from the database.
- `PromptService` — service layer exposing `get()`, `get_raw()`, `format()`, `render()`, and `list_keys()`. `render()` returns a fully-rendered `RenderedPrompt` (instructions + messages, ready to be adapted via `OpenAIMessageAdapter`).
- `@register_prompt` decorator — registers a `Prompt` at import time into the `PromptLoader`.
- `PromptDTO` / `ActivePromptDTO` — ORM models for versioned prompt storage with active-prompt tracking.

**Message templates:**

| Class | Role / Behaviour |
|-------|------------------|
| `SystemMessageTemplate` | `system` instruction |
| `DeveloperMessageTemplate` | `developer` instruction |
| `UserMessageTemplate` | `user` message |
| `AssistantMessageTemplate` | `assistant` message |
| `ToolMessageTemplate` | `tool` message (carries `tool_call_id`) |
| `MessagesPlaceholder` | Slot that fans out into a variable-length list of conversation messages at render time |

**`MessagesPlaceholder`** mirrors LangChain's placeholder of the same name: it reserves a position in a message-based template that gets expanded at render time into the list of messages stored under `variable_name` in the render kwargs. The expected shape under that key is a list of items each carrying `role` + `content` (and optionally `tool_call_id` for tool messages); `StoredMessage` objects work as-is and plain dicts are accepted too. Defaults to `optional=True` (missing / `None` → empty list); `system` / `developer` roles inside the placeholder are rejected because chat history is conversation, not instructions.

This is the canonical way to weave prior turns into a prompt: the system instructions, the placeholder, and the structured user message live in one typed template, and the agent just passes `history=` and `user_query=` into `render()`.

**Subclasser note.** Items implement `to_rendered_prompt_items(variables) -> list[RenderedPromptItem]` (plural). Most items return a single-element list; `MessagesPlaceholder` returns one entry per spliced message.

**Startup:** `PromptLoader().initialize()` is called during the FastAPI lifespan. It auto-imports all modules under `base/services/prompt/templates/`, triggering `@register_prompt` decorators, and optionally loads overrides from the database.

**Usage:**
```python
from base.models.view_models.prompt import (
    MessagesPlaceholder,
    Prompt,
    PromptTemplate,
    SystemMessageTemplate,
    UserMessageTemplate,
)
from base.services.prompt.decorator import register_prompt
from base.services.prompt.service import PromptService


@register_prompt
def my_prompt() -> Prompt:
    return Prompt(
        key="my_prompt",
        name="My Prompt",
        description="Answer a user question with prior turns in context.",
        template=PromptTemplate.from_messages([
            SystemMessageTemplate(content="You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            UserMessageTemplate(content="{user_query}"),
        ]),
    )


rendered = PromptService().render(
    "my_prompt",
    history=stored_messages,  # list[StoredMessage] from ChatHistoryService
    user_query="Hello!",
)
```

The resulting `RenderedPrompt` exposes `instructions` (system / developer blocks) and `messages` (user / assistant / tool turns, including the placeholder-expanded history). Pass it through `OpenAIMessageAdapter.to_responses_input(prompt=rendered)` (or the chat-completions equivalent) to assemble the final LLM input — no separate `messages=` argument needed when the template already splices history via a placeholder.

### Chat History

`ChatHistoryService` (`base/services/chat_history/__init__.py`) — `KeyedSingleton` keyed on `ChatHistoryConfig`. Owns canonical conversation message persistence on top of an `AbstractSequenceStore[ConversationId, StoredMessage]` backend, plus visibility filtering, tool-pair repair, and per-message token counting.

**Highlights:**
- **Per-conversation locking.** Sync (`threading.Lock`) and async (`asyncio.Lock`) lock maps serialize `next_sequence + append` per conversation so sequence assignment never races the SQL unique constraint. Sync and async paths are independent — pick one per conversation.
- **Token counting** via `TokenCounter` (tiktoken). Counted at append time and stored on the message; `tokenizer_model` comes from `ChatHistoryConfig`.
- **Visibility filter.** Messages with `metadata["is_policy_violated"] == True` are dropped from `get_messages` / `aget_messages` so they never re-enter an LLM prompt.
- **Tool-pair repair.** On fetch, ensures the LLM tool-call contract holds: orphan `tool` messages are dropped, and assistant turns whose tool calls don't all have matching tool results get demoted to plain assistant messages (tool calls stripped). Keeps the conversation submittable to OpenAI without bookkeeping at the call site.

**Public API:**

| Operation | Sync | Async |
|-----------|------|-------|
| Append a single message | `append_message` | `aappend_message` |
| Get messages (filtered + repaired) | `get_messages` | `aget_messages` |
| Clear all messages for a conversation | `clear` | `aclear` |

**Backing stores** live under `base/services/store/chat_history/`:
- `ChatHistoryStore` — marker subclass of `AbstractSequenceStore[ConversationId, StoredMessage]`. Service depends on this type, not on the abstract directly.
- `SqlChatHistoryStore` — SQLAlchemy-backed; one row per message in `chat_history_message`. Uses the `(conversation_id, sequence)` unique constraint as the correctness primitive.
- `InMemoryChatHistoryStore` — single-process reference impl for tests / local dev. IDs are not auto-assigned.

`StoredMessage` is the canonical, backend-neutral shape. SQL and Mongo (when implemented) round-trip through it; persistence timestamps stay on the DTO.

### Conversations

The conversations layer pairs a `ConversationStore` with a router and a chat-history store to provide a fully-wired chat surface:

- **`ConversationStore`** (`base/services/store/conversation.py`) — `AbstractStore[ConversationDTO]` with extra `*_for_user` variants (`aget_for_user`, `acreate_for_user`, `aedit_for_user`, `adelete_for_user`, …). Soft-deleting a conversation cascades a soft-delete to all its `ChatHistoryMessageDTO` rows in a single transaction.
- **`ConversationDTO` / `ChatHistoryMessageDTO`** ORM models — see [ORM Models](#orm-models).
- **Conversation router** (`base/routers/conversation.py`) — endpoints listed in [API Endpoints](#api-endpoints).

The router is also a worked example of how to assemble the streaming chat flow: it stamps the per-request debug-trace baseline (`execution_context.start_debug_trace()`), constructs a `FlowVariables`, populates the `ExecutionContext`, hands the work to a `TestAgent` instance (any `AbstractAgent` subclass), persists user/assistant turns through `ChatHistoryService`, and (in streaming mode) spawns the run as a background coroutine so the SSE response can drain independently.

### Background Tasks

`AbstractBackgroundTaskService` (`base/services/background_tasks/__init__.py`) — distributed-safe API. Tasks must be registered by name and invoked with pickleable args; each `enqueue` produces a unique `TaskHandle`. The interface is intentionally backend-agnostic so an in-memory implementation today and Celery / arq tomorrow share callsites.

**API:**

| Concern | Methods |
|---------|---------|
| Registration | `register(name, func, replace=False)`, `task(name)` decorator → `RegisteredTask` |
| Enqueue | `enqueue` / `aenqueue` |
| Cancel | `cancel`, `acancel`, `cancel_by_name`, `acancel_by_name` |
| Status / Result | `get_status` / `aget_status`, `get_result` / `aget_result(timeout=…)` |
| Lifecycle | `ashutdown(timeout=5.0)` |

`RegisteredTask[S]` is a typed reference returned by `register()` / `@task()`. It's directly callable (bypasses the queue) and exposes `enqueue` / `aenqueue` and `cancel_all` / `acancel_all` bound to the owning service.

**`InMemoryBackgroundTaskService`** (`base/services/background_tasks/in_memory.py`) — asyncio-based, in-process; state is lost on restart. Adds an extra `spawn_coroutine(coro, *, name=None, task_id=None, metadata=None)` escape hatch that accepts raw coroutines (e.g. closures over request-scope vars like a `StreamIterator` SSE pump). The escape hatch is in-process only; not portable to distributed backends.

**Usage:**
```python
from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService

service = InMemoryBackgroundTaskService()


@service.task("send_welcome_email")
async def send_welcome_email(user_id: str) -> None: ...


handle = await send_welcome_email.aenqueue(user_id)
await service.aget_result(handle, timeout=10.0)
```

### HTTP Client

`HttpRequestService` in `base/services/http/__init__.py` — a singleton HTTP client providing both sync (via `requests`) and async (via `httpx` with HTTP/2) request methods.

Features:
- Automatic retry with exponential backoff on timeouts
- JSON and non-JSON response parsing
- Error detail extraction
- Returns `HttpResponse` model with `ok()`/`fail()` factory methods

### Logging

`ConsoleLoggerService` (`base/services/logging/console.py`) — concrete singleton for colored console logging with structured error formatting (code, category, location, tag).

Logging config in `base/services/logging/config.py` uses `colorlog` for colored output.

---

## Agents (AbstractAgent)

`AbstractAgent` (`base/services/agents/__init__.py`) — template-method base for every LLM agent in the chat flow. One `SingletonABC` instance per concrete subclass; per-request state lives on the lazily-resolved `ExecutionContext`.

**Hooks** (all async, no arguments — agents read from / write to `self._execution_context.flow_variables`):

| Hook | Required | Purpose |
|------|----------|---------|
| `_pre_execute()` | No | Optional setup |
| `_execute()` | **Yes** | Main work; let `LLMError` propagate |
| `_post_execute()` | No | Optional teardown; runs only on success |

**Error mapping at `arun()`:** if `_execute()` raises an `LLMError`, `arun` looks up the matching `CannedMessages` (locale-aware via `flow_variables.locale`), sends it inline as a `StreamMessage` + `StreamEnd` when the request is streaming, and re-raises as `FlowAbortError(canned_message, cause=exc)` for the pipeline to catch. The pipeline / router converts that into the user-facing response — the request still completes normally from the client's point of view.

**Constants** subclasses set:
- `PROMPT_KEY: ClassVar[str]` — key into `PromptService` for the agent's system / developer prompt
- `MAX_OUTPUT_TOKENS: ClassVar[int]` — per-agent Responses API cap (default `1000`)
- `REASONING_EFFORT: ClassVar[ReasoningEffort | None]` — optional Responses API reasoning effort (`ReasoningEffort` in `base.models.definitions.model`); `None` omits the field
- `HISTORY_LIMIT: ClassVar[int | None]` — most recent N stored messages fetched for prompts (default `40`); `None` = unbounded

**Injected services** (set up in `AbstractAgent.__init__`): `OpenAIService`, `PromptService`, `ChatHistoryService`, `ConsoleLoggerService`. Pass `label=type(self).__name__` (or `f"{type(self).__name__}:{item_id}"` for fanout) to the LLM calls so each captured `DebugCall` carries a meaningful identifier.

**Built-in helpers:**
- `await self._aget_history()` — returns the conversation's `list[StoredMessage]` minus the trailing `USER` message, capped by `HISTORY_LIMIT`. The conversation router appends the current request's user message **before** the pipeline runs, so the latest stored `USER` is the current query; agent prompts already substitute that into their structured user message via `{user_query}`. Slicing it off here prevents duplicating the same text inline as a history entry. Pass the result as `history=` to `self._prompt_service.render(...)` and let a `MessagesPlaceholder` in the template splice it in. Returns `[]` when the context has no `conversation_id`.
- `await self._aresponses(input_, label=...)` — thin wrapper over `OpenAIService.aresponses` that applies `MAX_OUTPUT_TOKENS`, `REASONING_EFFORT`, streaming from `ExecutionContext`, and a default `label=type(self).__name__`. Explicit kwargs always win.

**Example** (see `base/services/agents/test.py`):
```python
from typing import ClassVar

from base.services.agents import AbstractAgent
from base.services.llm import OpenAIMessageAdapter


class TestAgent(AbstractAgent):
    PROMPT_KEY: ClassVar[str] = "test_prompt"
    MAX_OUTPUT_TOKENS: ClassVar[int] = 500

    async def _execute(self) -> None:
        ctx = self._execution_context
        rendered = self._prompt_service.render(
            self.PROMPT_KEY,
            history=await self._aget_history(),
            user_query=ctx.flow_variables.user_query,
        )
        result = await self._aresponses(
            OpenAIMessageAdapter.to_responses_input(prompt=rendered),
        )
        ctx.flow_variables.final_answer = (result.content or "").strip()
```

The companion prompt template uses a `MessagesPlaceholder` to receive the spliced history — see [Prompt Management](#prompt-management).

---

## Execution Context

`ExecutionContext` (`base/services/execution_context/__init__.py`) — per-request data container shared by every agent and service in the chat flow. Uses the `Stored` metaclass so each instance is bound to a request-scoped `scope_id` (set by the `register_request` middleware in [`base/main.py`](base/main.py)).

**Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `flow_variables` | `FlowVariables` | Single source of truth for one pipeline run (conversation id, user query, locale, final answer, your additions) |
| `streaming` | `bool` | Whether the current request is an SSE stream |
| `stream_iterator` | `StreamIterator \| None` | Per-request SSE pump (set by the router when `streaming=true`) |
| `debug_trace` | `DebugTrace` | Per-request bundle of `DebugCall`s + `FlowEvent`s, auto-populated by `OpenAIService` |

**Helpers:**
- `set_flow_variable(key, value)` / `get_flow_variable(key, default=None)` / `get_flow_variables()`
- `start_debug_trace()` — stamp `DebugTrace.started_at` (call once at the top of your pipeline so subsequent `record_event` calls have a baseline)
- `record_event(stage, **detail) -> FlowEvent` — append a breadcrumb (`elapsed_ms` is computed from `started_at`)

**Lifecycle.** The `register_request` middleware sets a fresh `scope_id` per HTTP request and calls `Stored.delete_scope()` after the response, except when the response is `text/event-stream` — SSE streams drain after the handler returns, so the scope persists for the duration of the stream.

Access pattern: `AbstractAgent` and `OpenAIService` resolve the context lazily through a property (`return ExecutionContext()`) instead of caching it in `__init__`, because both classes are singletons but the context is per-request scoped. Storing it once would freeze the first request's context across every subsequent request.

---

## Debug Trace

The debug trace is the per-request observability bundle on `ExecutionContext.debug_trace`. It's populated automatically — there's nothing to wire up beyond passing a meaningful `label=` to LLM calls.

**Captured shapes** (`base/models/view_models/debug_trace.py`):

- `DebugCall` — one LLM call: `agent`, `label` (e.g. `MyAgent` or `MyAgent:item_id`), `model`, `prompt`, `response`, `input_tokens`, `output_tokens`, `total_tokens`, `duration_ms`, `error`. `OpenAIService` writes one of these around every `completions` / `responses` / `parse_completions` / `parse_responses` call (and their async siblings) via a `try/finally`, so failed calls are recorded too.
- `FlowEvent` — one pipeline-level breadcrumb: `elapsed_ms`, `stage`, `detail`. Use `ExecutionContext.record_event("identify_path", chosen=…)` to leave them.
- `DebugTrace` — `started_at` + `calls` + `events`. Persisted to `chat_history_message.metadata["debug"]` when the conversation router writes the assistant turn (see [`base/routers/conversation.py`](base/routers/conversation.py)).

A trace is only populated inside a pipeline run (`debug_trace.started_at != 0`). Outside of one (standalone LLM calls, admin routes), `OpenAIService` still logs a one-liner per call but skips the trace appendage.

---

## Localization & Canned Messages

The starter ships a small bilingual (English / Arabic) localization layer used by the agent error path.

- **`Locale`** (`base/models/definitions/database.py`) — `EN`, `AR`, `OTHER`. Used as a dict key in `CannedMessages` values.
- **`get_text_locale(text, fallback_text=None, default_fallback=None)`** (`base/tools/locale.py`) — counts Latin vs. Arabic-script glyphs to pick a `Locale` for the user's input. Falls back to `default_fallback` (or `Locale.EN`) when the text contains neither (numbers / punctuation only).
- **`CannedMessages`** (`base/models/definitions/canned_messages.py`) — enum where each value is a `dict[Locale, str]`. Includes `ERROR_FALLBACK`, `LLM_GENERIC_ERROR`, `LLM_RATE_LIMITED`, `LLM_INCOMPLETE`, `LLM_STREAM_FAILED`, `JAILBREAK_ATTEMPT`, and per-category content-filter messages (hate, violence, sexual, self-harm).
- **`CannedMessages.from_llm_error(exc)`** — maps an `LLMError` to the matching message. For `LLMContentFilterError`, inspects `incomplete_details.content_filter_results` and picks the specific category (hate / violence / sexual / self-harm / jailbreak), falling back to `JAILBREAK_ATTEMPT`.

Pattern in agents: `AbstractAgent.arun` calls `CannedMessages.from_llm_error(exc).value[locale]` to surface a localized, user-safe message — never the raw provider error.

---

## Error Handling

`AppExceptionHandler` (`base/tools/exception_handlers.py`) registers FastAPI handlers for the four error tiers and produces consistent `ErrorResponse[T]` payloads. Registered once in `base/main.py` via `AppExceptionHandler().register(app)`.

| Trigger | Handler | Status |
|---------|---------|--------|
| `RequestValidationError` | `handle_request_validation_error` | 422 |
| `StarletteHTTPException` | `handle_starlette_http_exception` | passthrough |
| `ErrorsException` (`raise_error(...)` from `base/tools/errors.py`) | `handle_errors_exception` | from enum or arg |
| Unhandled `Exception` | `handle_unhandled_exception` | 500 |

**Debug context.** Outside `ENVIRONMENT == "production"`, every error response includes a `debug: DebugContext` block with `request_id` (from `scope_id`), `path`, `method`, and (for unhandled exceptions) the formatted traceback. In production, `error` collapses to `"Internal server error."` and traceback is omitted.

**Logging.** 5xx errors and unhandled exceptions are logged at error level with `category=ExceptionCategory.API` / `INTERNAL` and the request path; 4xx errors are logged at warning level.

Domain LLM errors (the `LLMError` hierarchy) are caught at the agent / pipeline boundary — they never bubble up to `AppExceptionHandler`.

---

## Authentication

JWT-based authentication with RS256 (RSA asymmetric keys), optional social Login-with-X, and optional enterprise IdP resource-server mode.

**Components:**
- `AuthService` (`base/services/auth/__init__.py`) — local token mint/refresh/validate; optional IdP Bearer validation via JWKS
- `OAuthService` (`base/services/auth/oauth/`) — OAuth/OIDC authorize + callback + exchange for Google / Facebook / Microsoft / GitHub / Apple
- `user_identity` table — links `(provider, subject)` → local `user.id` (stable app UUID for FKs)
- `PasswordHashingService` (`base/tools/security.py`) — Argon2 (password accounts only; social-only users have `password_hash=null`)
- `avalidate_user` / `validate_user_dependency` — route protection

**Local password flow:**
1. Register: hash password, create user
2. Login: verify password, mint access + refresh token pair
3. Protected routes: validate access token → `UserDTO`
4. Refresh: validate refresh token, mint new pair

**Social Login-with-X (env-gated):**
1. Enable providers: `OAUTH_PROVIDERS=google,github,apple` plus each provider’s credentials / redirect URI
2. Browser: `GET /auth/oauth/{provider}` → redirect to IdP
3. Callback: `GET|POST /auth/oauth/{provider}/callback` → upsert `user` + `user_identity` → redirect to `OAUTH_SUCCESS_REDIRECT?code=…` (Apple uses `form_post`)
4. SPA: `POST /auth/oauth/exchange` with `{ "code": "…" }` → local `TokenPair`
5. List enabled providers: `GET /auth/oauth/providers`

**Sign in with Apple:** set `OAUTH_APPLE_CLIENT_ID` (Services ID), `OAUTH_APPLE_TEAM_ID`, `OAUTH_APPLE_KEY_ID`, and `OAUTH_APPLE_PRIVATE_KEY_PATH` (or inline PEM). The API mints the ES256 client-secret JWT at token exchange. Optionally set a pre-generated `OAUTH_APPLE_CLIENT_SECRET` instead.

Password and social coexist. Verified-email auto-link attaches a new provider to an existing local user when the provider email is verified.

**Enterprise IdP resource server (optional):**
Set `IDP_RESOURCE_SERVER_ENABLED=true` plus `IDP_ISSUER`, `IDP_AUDIENCE`, and `IDP_JWKS_URL` (or discovery). Clients may send the IdP access token as `Authorization: Bearer …`; the API verifies JWKS/`iss`/`aud`, maps via `user_identity` (`IDP_PROVIDER_NAME`), and returns the same `UserDTO` contract. Local app JWTs still work.

**Token types:** `AccessToken` and `RefreshToken` are `NewType` wrappers around `str` for type safety.

---

## Store (Repository) Layer

Two complementary abstract bases live in `base/services/store/__init__.py`:

- **`AbstractStore[T]`** — row-keyed entity CRUD (the classic repository pattern).
- **`AbstractSequenceStore[TParent, TItem]`** — ordered items under a parent key (chat history, event log, notifications). Sibling to `AbstractStore`, not a subclass.

### AbstractStore[T]

A **database-agnostic** generic repository pattern with complete CRUD + pagination. Works with both SQL and NoSQL backends. All operations have sync and async variants:

| Operation | Sync | Async |
|-----------|------|-------|
| Get by ID/column | `get()` | `aget()` |
| Get many | `get_many()` | `aget_many()` |
| Add | `add()` | `aadd()` |
| Add many | `add_many()` | `aadd_many()` |
| Edit | `edit()` | `aedit()` |
| Delete (soft) | `delete()` | `adelete()` |
| Get all | `get_all()` | `aget_all()` |
| Get all paged | `get_all_paged()` | `aget_all_paged()` |

Public methods are `@final` and handle duplicate-key errors → `UNIQUE_CONSTRAINT_VIOLATION` and missing records → `RECORD_NOT_FOUND`. Subclasses implement the `_` prefixed abstract methods.

`AbstractStore` has **no dependency on any specific database library**. Subclasses are responsible for:
- Setting up their own database service (SQL or NoSQL) in `__init__`
- Implementing all abstract CRUD methods using their chosen service
- Overriding `_is_duplicate_error(error)` to map their DB driver's duplicate-key exception

### AbstractSequenceStore[TParent, TItem]

Use this instead of `AbstractStore` when the data is grouped under a parent entity, ordered by a monotonic per-parent `sequence: int`, appended far more often than edited, and sometimes cleared or range-replaced as a group.

| Operation | Sync | Async |
|-----------|------|-------|
| Append items | `append()` | `aappend()` |
| Get items (optionally last `limit`) | `get()` | `aget()` |
| Count items | `count()` | `acount()` |
| Next sequence number | `next_sequence()` | `anext_sequence()` |
| Clear all items for a parent | `clear()` | `aclear()` |
| Replace items up to a sequence | `replace_range()` | `areplace_range()` |

Concrete sub-types live per backend (SQL, Mongo, in-memory) and per domain (chat history, event log, notifications). The chat-history pair is a working example: `ChatHistoryStore` is a marker subclass that names the contract (`AbstractSequenceStore[ConversationId, StoredMessage]`), and `SqlChatHistoryStore` / `InMemoryChatHistoryStore` are the concrete implementations. See [Chat History](#chat-history).

### SQL Store Example (UserStore)

```python
class UserStore(AbstractStore[UserDTO], metaclass=SingletonABC):
    def __init__(self) -> None:
        self._sql_service = SQLService()
        self._base_select_query = select(UserDTO)

    def _is_duplicate_error(self, error: Exception) -> bool:
        return isinstance(error, IntegrityError)

    def _get(self, value: Any, column_name: str = "id") -> UserDTO | None:
        query = self._base_select_query.where(getattr(UserDTO, column_name) == value)
        return self._sql_service.query(query, SQLResultFormats.one_or_none_scalar)
    ...
```

### NoSQL Store Example (MongoDB)

```python
class DocumentStore(AbstractStore[dict[str, Any]], metaclass=SingletonABC):
    def __init__(self) -> None:
        super().__init__()
        self._nosql_service = MongoDBService()
        self._collection_name = "documents"

    def _is_duplicate_error(self, error: Exception) -> bool:
        from pymongo.errors import DuplicateKeyError
        return isinstance(error, DuplicateKeyError)

    def _get(self, value: Any, column_name: str = "id") -> dict[str, Any] | None:
        return self._nosql_service.find_one(self._collection_name, {column_name: value})
    ...
```

---

## Models

### ORM Models

Located in `base/models/orm/`.

- `Base` — `MappedAsDataclass + DeclarativeBase` with `to_dict()`, `update_from_dict()`, `clone()` helpers
- `TimestampMixin` — adds `created_at`, `updated_at`, `deleted_at` with `soft_delete()` and `restore()` methods
- `UserDTO` — user model with UUID primary key, email, nullable `password_hash` (null for social-only), name fields, and indexes
- `UserIdentityDTO` — links external `(provider, subject)` to `user.id` (unique per provider+subject)
- `PromptDTO` — versioned prompt template storage with key + version unique constraint
- `ActivePromptDTO` — tracks the currently active prompt version per key (FK to `PromptDTO`)
- `ConversationDTO` — chat conversation owned by a user (UUID PK, `user_id` FK, `title`, `meta` JSON, soft-delete + indexes on `user_id` and `created_at`)
- `ChatHistoryMessageDTO` — one row per message under a conversation; UUID PK, `conversation_id` FK, monotonic `sequence`, `role` (enum), `content`, `tool_calls` (JSON), `tool_call_id`, `token_count`, `meta`. Unique constraint on `(conversation_id, sequence)` enforces ordering correctness.

ORM models are auto-discovered via `pkgutil.walk_packages` in `base/models/orm/__init__.py`.

### View Models (Pydantic)

Located in `base/models/view_models/`.

**Core wrappers:**
- `BaseModel` — Pydantic BaseModel with camelCase alias generation, `serialize()`, `deserialize()`, `jsonify()` methods
- `BaseResponse[T]` — generic success response wrapper (`success: True, data: T`)
- `ErrorResponse[T]` — generic error response wrapper (`success: False, error: str, debug: DebugContext | None`)
- `DebugContext` — request id, path, method, optional traceback (populated outside production by `AppExceptionHandler`)
- `HttpResponse` — HTTP response model with `ok()`/`fail()` factory methods

**Pagination:**
- `PaginatedResult[T]` — paginated response with `data`, `page`, `size`, `total`, `total_pages`
- `PaginationParams`, `SortingParams`, `FilteredPaginationParams` — query parameter models

**Streaming:**
- `StreamStart`, `StreamMessage`, `StreamEnd`, `StreamError` — SSE stream event models

**LLM:**
- `ModelConfigurations` — LLM provider configuration model
- `EmbeddingConfigurations` — text embedding provider configuration model
- `OpenAIModels` — OpenAI model name enum (used by `TokenCounter`, etc.)
- `LLMError` and subclasses — `LLMConfigError`, `LLMBadRequestError`, `LLMAuthError`, `LLMNotFoundError`, `LLMConflictError`, `LLMRateLimitError`, `LLMServerError`, `LLMTimeoutError`, `LLMConnectionError`, `LLMContentFilterError`, `LLMIncompleteError`, `LLMStreamError`, `LLMRetryExhausted`, `LLMUnexpectedError`
- `CompletionResult`, `ResponseResult`, `ParsedCompletionResult[T]`, `ParsedResponseResult[T]` — return shapes from `OpenAIService`, with `.content` and `.tool_calls` helpers

**Prompts (subpackage `prompt/`):**
- `Prompt` — `key`, `name`, `description`, `template`, with `render(**vars)` returning a `RenderedPrompt`
- `PromptTemplate` — accepts either a `template: str` or `messages: list[PromptMessageTemplate]`; validates declared/detected variables; supports `partial_variables`; factory methods `from_messages` / `from_template`
- `BaseMessageTemplate`, `SystemMessageTemplate`, `DeveloperMessageTemplate`, `UserMessageTemplate`, `AssistantMessageTemplate`, `ToolMessageTemplate`
- `MessagesPlaceholder` — fan-out template item that splices a list of conversation messages (e.g. chat history) into a typed prompt at render time; `variable_name`, `optional` (default `True`)
- `RenderedPrompt`, `RenderedPromptInstruction`, `RenderedPromptMessage`, `RenderedPromptItem`

**Chat & conversations:**
- `ConversationId` — `NewType` over `UUID`, used as the `TParent` for chat-history sequence stores
- `StoredMessage` — canonical conversation message: `id`, `conversation_id`, `sequence`, `role`, `content`, `tool_calls`, `tool_call_id`, `token_count`, `metadata` (validated by role)
- `StoredToolCall`, `StoredToolCallFunction` — function-type tool calls
- `ChatHistoryConfig` — config for `ChatHistoryService` (just `tokenizer_model` for now)
- `ChatMessageView`, `SendChatRequest`, `SendChatResponse`, `ListChatResponse` — chat router I/O
- `CreateConversationRequest/Response`, `GetConversationResponse`, `EditConversationRequest/Response`, `DeleteConversationResponse` — conversation router I/O

**Flow & execution context:**
- `FlowVariables` — single source of truth for one pipeline run (`conversation_id`, `user_query`, `locale`, `final_answer`, plus whatever your pipeline adds)
- `FlowAbortError` — raised by `AbstractAgent` on a fatal LLM error; carries a `CannedMessages` for the pipeline to surface

**Debug trace:**
- `DebugCall` — one captured LLM call (agent, label, model, prompt, response, token counts, duration, error)
- `FlowEvent` — one pipeline-level breadcrumb (`elapsed_ms`, `stage`, `detail`)
- `DebugTrace` — `started_at`, `calls`, `events`. Auto-populated by `OpenAIService` and persisted to chat-history `metadata.debug`

**Background tasks:**
- `TaskHandle[T]` — `id`, `name`, `backend`, `created_at`, `metadata`

**Auth & users:**
- `RegisterRequest`, `LoginRequest/Response`, `RefreshRequest/Response`, `TokenPair`
- `OAuthExchangeRequest`, `OAuthProvidersResponse`
- `GetUserResponse`, `EditUserRequest/Response`, `CreateUserResponse`, `DeleteUserResponse`

### Definitions (Enums, Constants)

Located in `base/models/definitions/`.

- `AuthErrors`, `DatabaseErrors`, `GenericErrors` — error enums with message and status code
- `SQLBackend` — SQL backend enum (`sqlite`, `mysql`, `postgresql`) with `sync_driver`/`async_driver` properties
- `DatabaseTable` — table name constants (`user`, `user_identity`, `prompt`, `active_prompt`, `conversation`, `chat_history_message`)
- `Locale` — `EN`, `AR`, `OTHER` (text locale, used by `CannedMessages` and `get_text_locale`)
- `SortingOrder` — `ASC`, `DESC`
- `CacheChannels` — Redis pub/sub channel names (`test`, `keyvault_reload`, `logs`, `prompt_reload`)
- `ProviderType` — AI provider enum (OpenAI, Azure OpenAI, Ollama, vLLM, Groq, Together, LiteLLM)
- `MessageRole` — chat message role enum (`system`, `developer`, `user`, `assistant`, `tool`)
- `CannedMessages` — bilingual user-facing strings keyed by `Locale`; `from_llm_error(exc)` maps an `LLMError` (rate limit, content filter category, incomplete, stream, generic) to the matching message
- `TaskStatus` — background-task lifecycle (`pending`, `running`, `success`, `failed`, `cancelled`)
- `BackgroundTaskBackend` — `in_memory`, `celery`, `arq`
- `ExceptionCategory` — exception classification
- `DistanceMetric` — vector search distance metrics

---

## Routers

Routers are auto-discovered from the `base/routers/` package using `pkgutil.iter_modules`. Each module must export a `router` variable (`APIRouter` instance).

All routers are mounted under the `/api/7r` prefix.

**Included routers:**

- `auth.py` — `/auth` register, login, refresh, oauth providers/start/callback/exchange
- `user.py` — `/users/me` get, edit, soft-delete
- `conversation.py` — `/conversations` create, get, edit, delete; `/conversations/{id}/messages` send (optionally streaming) + list

`auth.py` and `user.py` are minimal reference implementations; `conversation.py` is a complete end-to-end example showing how to wire `ConversationStore`, `ChatHistoryService`, `ExecutionContext`, an `AbstractAgent` subclass (`TestAgent`), `InMemoryBackgroundTaskService`, and `StreamIterator` together. Use it as the template for your own conversational endpoints.

### Adding a New Router

1. Create a file in `base/routers/` (e.g., `items.py`)
2. Define `router = APIRouter(prefix="/items", tags=["Items"])`
3. Add your route handlers
4. It's automatically registered — no manual wiring needed

---

## Tools (Utilities)

| Module | Purpose |
|--------|---------|
| `database.py` | `SQLResultFormats` — query result processor functions for SQLAlchemy |
| `errors.py` | `ErrorsException`, `StateDecodeError`, and `raise_error()` helper |
| `exception_handlers.py` | `AppExceptionHandler` — registers FastAPI handlers for `RequestValidationError`, `StarletteHTTPException`, `ErrorsException`, and unhandled `Exception`; injects `DebugContext` outside production |
| `security.py` | `PasswordHashingService` (Argon2 hashing) |
| `hashing.py` | `hash_args(*args, **kwargs)` — deterministic SHA-256 of arbitrary args via `ExpansiveEncoder` |
| `serializers.py` | `ExpansiveEncoder` (JSON encoder for Pydantic, DTOs, UUIDs, datetimes), `serialize()` |
| `deserializers.py` | JSON deserialization helpers |
| `routers.py` | `init_routers()` — auto-discovery router registration |
| `webserver.py` | `run_webserver()` — Uvicorn (dev) / Gunicorn (prod) launcher |
| `stream.py` | `StreamIterator` — async iterator for SSE streaming |
| `tokenizer.py` | `TokenCounter` — `KeyedSingleton` tiktoken wrapper, falls back to `cl100k_base` for unknown models |
| `prompt.py` | `extract_template_variables` (root-name extraction from `str.format` placeholders), `safe_format` (no-op when no `{` present — protects opaque IDs) |
| `validators.py` | Password validation logic |
| `misc.py` | JSON serialization, general utilities |
| `locale.py` | `get_text_locale(text, fallback_text=None, default_fallback=None)` — EN/AR detection over Unicode ranges |

---

## Streaming (SSE)

The starter includes full Server-Sent Events support via `sse-starlette`.

**Components:**
- `StreamIterator` (`base/tools/stream.py`) — async iterator that wraps an `asyncio.Queue` or `AsyncIterator` and formats messages as SSE events
- Stream models (`base/models/view_models/streams.py`):
  - `StreamStart` — sent when streaming begins
  - `StreamMessage` — carries content chunks
  - `StreamEnd` — signals stream completion
  - `StreamError` — signals stream failure
- `OpenAIService` integrates directly with `StreamIterator` for streaming LLM responses

---

## Live Log Streaming

The starter includes a built-in live log streaming system via `TerminalCaptureService`:

1. At startup, `TeeStream` wraps `stdout`/`stderr` to capture all output
2. Captured output is buffered and published to Redis pub/sub
3. The `POST /logs` endpoint (authenticated) returns an SSE stream of live application logs

This enables real-time log monitoring from the frontend without direct server access.

---

## Webserver

The webserver launcher (`base/tools/webserver.py`) automatically selects the server based on environment:

| Environment | Server | Features |
|-------------|--------|----------|
| `local` | Uvicorn | Hot reload enabled |
| Any other | Gunicorn + Uvicorn workers | Multi-process, configurable workers, preload |

Gunicorn is configured with:
- Worker count: `min(cpu_count + 1, max_gunicorn_workers)`
- Worker class: `UvicornWorker`
- Timeout: 500s, graceful timeout: 300s

---

## API Endpoints

### Main

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Welcome message + version |
| GET | `/health-check` | No | Health check |
| POST | `/logs` | Yes | Live SSE log stream |

### Auth (`/api/7r/auth`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/register` | No | Create password user account |
| POST | `/login` | No | Password login, get token pair |
| POST | `/refresh` | No | Refresh local token pair |
| GET | `/oauth/providers` | No | List enabled OAuth providers |
| GET | `/oauth/{provider}` | No | Start Login-with-X (302 to provider) |
| GET\|POST | `/oauth/{provider}/callback` | No | OAuth callback (Apple uses POST/`form_post`) |
| POST | `/oauth/exchange` | No | Trade short-lived `code` for local token pair |

### Users (`/api/7r/users`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/me` | Yes | Get current user profile |
| PATCH | `/me` | Yes | Edit profile and/or change password (`currentPassword` + `newPassword`) |
| DELETE | `/me` | Yes | Soft-delete current user |

### Conversations (`/api/7r/conversations`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/` | Yes | Create a conversation for the current user |
| GET | `/{conversation_id}` | Yes | Get a conversation owned by the current user |
| PATCH | `/{conversation_id}` | Yes | Edit a conversation's title / metadata |
| DELETE | `/{conversation_id}` | Yes | Soft-delete a conversation (cascades to its messages) |
| POST | `/{conversation_id}/messages` | Yes | Send a chat message; pass `?streaming=true` for SSE |
| GET | `/{conversation_id}/messages` | Yes | List canonical messages for a conversation |

`POST /{conversation_id}/messages?streaming=true` returns `text/event-stream` (`StreamStart` → `StreamMessage` deltas → `StreamEnd`, with `StreamError` on failure). The handler runs the chat pipeline as an `InMemoryBackgroundTaskService.spawn_coroutine` task so the stream drains independently of the request handler.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.12+ |
| **Framework** | FastAPI + Uvicorn (standard) + Gunicorn |
| **ORM** | SQLAlchemy 2.x (async + sync) |
| **Validation** | Pydantic v2, pydantic-settings |
| **Auth** | PyJWT (RS256), Argon2, OAuth/OIDC Login-with-X, optional IdP JWKS |
| **HTTP Client** | httpx (HTTP/2), requests, aiohttp |
| **Databases** | SQLite, MySQL, PostgreSQL, MongoDB, Azure Cosmos DB |
| **Cache** | Redis (pub/sub, caching, lists) |
| **Vector DBs** | Qdrant, Azure AI Search, MongoDB Atlas Search |
| **LLM** | OpenAI SDK (Chat Completions + Responses APIs; OpenAI, Azure OpenAI, any compatible) |
| **Prompt Management** | Custom `PromptTemplate` (no LangChain dependency) |
| **Tokenizer** | tiktoken (per-model encoding, `cl100k_base` fallback) |
| **Cloud** | Azure Key Vault, Azure Blob Storage, Azure Cosmos DB, Azure Identity |
| **Streaming** | SSE via sse-starlette |
| **Password Hashing** | argon2-cffi |
| **Logging** | colorlog (colored console output) |
| **Package Manager** | uv |
| **Linting** | Ruff |
| **Pre-commit** | pre-commit hooks |

---

## Multi-Service Architecture (design)

This starter currently targets a **single application**. A design for extending it into a
**multi-app / microservice** starter — several deployables sharing this same foundation,
each with its own dependencies, its own database, and its own `.env` — lives in
[`docs/architecture/`](docs/architecture/README.md).

> **Design only — not yet implemented.** Nothing in the repository has changed.

| Doc | Covers |
|-----|--------|
| [Overview](docs/architecture/README.md) | Decisions at a glance |
| [01 — Workspace Layout](docs/architecture/01-workspace-layout.md) | uv workspace, one lockfile, many packages |
| [02 — Library Boundary](docs/architecture/02-library-boundary.md) | What stays in `base`, dependency extras, app factory |
| [03 — Data Ownership](docs/architecture/03-data-ownership.md) | Database per service, SQL/NoSQL freedom, per-service migrations |
| [04 — Configuration](docs/architecture/04-configuration.md) | `CoreSettings` + mixins, per-service `.env`, precedence |
| [05 — Service Contracts](docs/architecture/05-service-contracts.md) | Auth split, service-to-service calls, correlation IDs |
| [06 — Build & Deploy](docs/architecture/06-build-and-deploy.md) | One parameterised Dockerfile, compose, CI |
| [07 — Modularity Backlog](docs/architecture/07-modularity-backlog.md) | Remaining gaps, ranked |
| [08 — Migration Plan](docs/architecture/08-migration-plan.md) | Phased path; the app stays deployable throughout |
