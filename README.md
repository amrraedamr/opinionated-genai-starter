# Opinionated GenAI Starter

[![CI](https://github.com/amrraedamr/opinionated-genai-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/amrraedamr/opinionated-genai-starter/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A FastAPI foundation for LLM products that runs as one service or several,
from one codebase.** A uv workspace with a shared `base` package and whatever
services you need — one to start, more when a part of the system earns its own
database, scaling and release cycle.

You do not have to choose up front. The same slices — auth, conversations,
event ingestion — compose into a single deployment or split across services:

```bash
# One service that issues tokens and serves chat
uv run python tools/new_service.py --name app --from identity,chat_api

# Or keep them apart; that is what the reference services already do
uv run python tools/new_service.py --name billing --storage postgres --http
```

The hard part is not splitting the code; it is stopping the split from leaking.
So the boundaries here are enforced by lints that fail CI, not by a convention
everyone agrees to in a meeting and forgets by the second sprint.

---

## The shape

```
packages/
  base/          shared library: LLM, auth, stores, cache, vectors, RAG, broker
  contracts/     versioned event payloads every service agrees on
services/
  identity/      users, registration, OAuth, token issuing      → identity_db
  chat_api/      conversations, messages, agents                → chat_api_db
  ingestion_worker/  consumes events, durable inbox               → worker_db
tools/           scaffolder, boundary lint, affected detection
tests/           unit, integration and live suites
deploy/          compose stacks
docker/          one parameterised image for every service
```

Three services ship as reference implementations, not as the product. They
are also the test matrix: a worker that serves no HTTP is what keeps HTTP
concerns out of `base`, and an auth service is what keeps token issuance
separable from verification. `tools/new_service.py` writes the fourth, either
fresh or composed from the slices these three already prove.

---

## The rules, and what enforces them

`uv run python tools/boundary_lint.py` fails CI on any of these:

1. The shared `base` package imports no service.
2. No service imports another service.
3. `base` declares no table and no declarative root.
4. Each SQL service declares exactly one declarative root.
5. No foreign key crosses a service boundary.

The last one is the load-bearing rule. **A service owns its database.** Not a
schema inside a shared database — its own. If `chat_api` needs a user, it
holds an opaque id and asks `identity`, because a foreign key would quietly
turn two services back into one.

Settings work the same way. `CoreSettings` holds only what every service truly
shares — name, environment, log level — and everything else is a mixin a
service opts into:

```python
class Settings(CoreSettings, SQLSettingsMixin, PostgresSettingsMixin,
                TokenVerificationMixin, LLMSettingsMixin,
                AzureOpenAILLMSettingsMixin):
    ...
```

`chat_api` composes `TokenVerificationMixin`, so it can check a token but
cannot mint one — it has no private key setting to misuse. `identity`
composes `TokenIssuingMixin`. The boundary is in the type, not the README.

Dependencies follow: each service declares the extras it needs, so the
worker's image has no web server in it.

---

## Running it

```bash
uv sync --all-packages
docker compose -f deploy/docker-compose.infra.yml up -d   # Postgres, Redis, Mongo, Qdrant
```

Then one service, from its own directory so its `.env` resolves:

```bash
cd services/chat_api
cp .env.example .env
uv run alembic upgrade head
uv run python -c "from chat_api.app import start; start()"
```

Or the whole stack:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Every service builds from the same image, selected by a build argument:

```bash
docker build -f docker/service.Dockerfile --build-arg SERVICE=identity -t identity .
```

The image contains only the service it was built for; the others are removed
before the runtime stage.

---

## Adding a service

```bash
uv run python tools/new_service.py --name billing --storage postgres --http
uv lock
```

That writes the whole skeleton — settings composed from the mixins that
storage choice implies, an app factory, Alembic wiring, an `.env.example` —
and CI picks it up automatically, because the build matrix is derived from
what changed:

```bash
uv run python tools/affected.py --base origin/main
```

### Composing slices into one service

A *slice* is a reference service's domain code — its routers, ORM models, view
models and stores. `--from` copies those into a new service, rebinding them
onto a single declarative root:

```bash
uv run python tools/new_service.py --name app --from identity,chat_api
```

That yields one deployment serving `/auth/login`, `/users/me` and
`/conversations`, with one database, one outbox and one process. The settings
class is the union of each slice's mixins, the extras are the union of their
dependencies, and `create_app` receives every capability and lifespan hook the
slices declared.

Each service describes its own slice in `[tool.starter.slice]`, and membership
is by exclusion: every module travels except the composition roots the
generator rewrites, so a file you add to a reference service joins its slice
automatically. Generation refuses to proceed on a colliding table or router
module rather than emitting something that breaks at migration time.

The reference services are the only copy of this code — there is no second
template to drift. CI generates a composed service on every run and asserts it
lints, builds, serves both slices' routes and merges to one outbox.

---

## What the shared base gives you

Shared infrastructure, available to every service whether you run one or many:

- **Auth** — RS256 access and refresh tokens, Argon2 hashing, OAuth providers,
  and an enterprise IdP mode that validates someone else's tokens via JWKS.
- **LLM** — provider SDKs directly, not LangChain. `AbstractLLMService` gives
  one call shape; OpenAI (chat completions and the Responses API) and Anthropic
  sit behind it, each in its own package under `base.services.llm`. Streaming,
  structured output, tool calls and multimodal content work the same whoever
  answers, and error handling maps provider failures to one hierarchy your code
  can branch on.
- **Tool calling** — a tool is a function plus the Pydantic model describing
  its arguments; that model generates the schema the LLM sees and validates
  what comes back. `ToolRunner` executes the calls and loops until the model
  answers, against any provider behind the LLM seam.
- **RAG** — chunk, embed, upsert, search, with the passage stored next to its
  vector. Re-ingesting replaces rather than accumulating stale chunks.
  Retrieval fuses vector and full-text hits by rank where the backend offers
  both, and will rerank through a cross-encoder when one is supplied. Both are
  optional and off unless you pass them.
- **Usage** — every model call recorded at the service boundary, totalled per
  request, priced in `Decimal`. `UsageTracker().totals` answers "what did that
  cost". Prompt-cache reads and writes are counted separately from fresh input,
  because providers discount the first and charge a premium for the second, so
  a single input figure cannot be reconciled against an invoice.
  `totals.cache_hit_rate` is also where a prefix-stability regression shows up
  first: the rate collapses while the token counts stay the same.
- **Stores** — `AbstractStore[T]` and `AbstractSequenceStore`, one shared SQL
  implementation, and a conformance suite so a new backend is verified rather
  than assumed.
- **Infrastructure** — Redis or in-memory cache, Mongo, Cosmos, Azure Blob,
  Key Vault, and vector search over pgvector, Qdrant or Azure AI Search.
  pgvector needs no extra dependency: it runs on the Postgres a SQL service
  already has, so `base[sql,postgres]` is enough.
- **Input hygiene** — stored conversation turns are stripped of Unicode tag
  characters and other invisible text before they reach a model, closing one
  narrow and deterministic class of prompt injection. Joiners, variation
  selectors and bidirectional controls are deliberately left alone, because
  they are load-bearing for emoji and for Arabic.

---

## How services talk

Not by reaching into each other's data. Through versioned events on a broker:

```python
# Inside the owning service transaction:
from base.services.messaging import add_event

session.add(user)
add_event(session, OutboxEventDTO, Event.of(
    identity.USER_REGISTERED,
    UserRegistered(user_id=user.id, email=user.email, registered_at=user.created_at),
    producer="identity",
))
```

`packages/contracts` holds the payload every side agrees on, versioned so a
producer can move without breaking a consumer. The broker is an abstraction
with an in-memory implementation for local development and Redis Streams for
real use, with consumer groups and at-least-once delivery. The worker persists inbox claims keyed by `event.meta.event_id`, so idempotency survives process restarts.

A correlation id is adopted from `X-Correlation-ID` or a W3C `traceparent`,
echoed on the response, and carried through the broker onto the worker, so one
logical request is traceable across the services it touches.

---

## Working on it

```bash
uv run ruff check .
uv run python tools/boundary_lint.py
```

`CONTRIBUTING.md` covers the rest.

---

## Design notes

`docs/architecture/` records why the structure is what it is: the library
boundary, data ownership, configuration composition, service contracts, build
and deploy, and the migration plan that got here.

---

## Licence

MIT. See [LICENSE](LICENSE).

## Production hardening

Registration and conversation creation write their domain row and an outbox row in the same service-owned transaction. Per-service dispatchers publish committed rows to Redis Streams and persist retry state; request handlers never publish directly. The ingestion worker owns a small SQL inbox that commits claims and terminal handler state, so broker redelivery is idempotent across restarts. Redis consumers reclaim idle pending entries, back off bounded retries, and move exhausted deliveries to `<topic>.dlq` with failure metadata.

```bash
uv run broker-ops replay-dlq identity.user.registered --limit 100
```

Non-local logging is structured JSON with correlation IDs and redaction. Prompt/response content is not traced or persisted in normal chat history by default. Compose is local only; see `docs/architecture/06-build-and-deploy.md` for what a production deployment has to satisfy. In-process limits are per replica; use an external gateway and Redis-backed/global controls for fleet-wide enforcement.
