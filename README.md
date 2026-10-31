# Opinionated GenAI Microservices Starter

[![CI](https://github.com/amrraedamr/opinionated-genai-microservices-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/amrraedamr/opinionated-genai-microservices-starter/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Several FastAPI services for LLM products, sharing one foundation.** A uv
workspace with a shared `base` package and independently deployable services,
each owning its own database, settings, dependencies and release.

The hard part of microservices is not splitting the code; it is stopping the
split from leaking. So the boundaries here are enforced by a lint that fails
CI, not by a convention everyone agrees to in a meeting and forgets by the
second sprint.

> Building one application rather than several? Start with the single-app
> variant instead: [opinionated-genai-starter](https://github.com/amrraedamr/opinionated-genai-starter).
> It is the same foundation in one deployable, and features port across.

---

## The shape

```
packages/
  base/          shared library: LLM, auth, stores, cache, vectors, RAG, broker
  contracts/     versioned event payloads every service agrees on
services/
  identity/      users, registration, OAuth, token issuing      → identity_db
  chat_api/      conversations, messages, agents                → chat_api_db
  ingestion_worker/  consumes events, no HTTP surface           → no database
tools/           scaffolder, boundary lint, affected detection, test runner
deploy/          compose stacks
docker/          one parameterised image for every service
```

Three services ship as examples, not as the product. `tools/new_service.py`
writes the fourth.

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
                TokenVerificationMixin, LLMSettingsMixin):
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

---

## What the shared base gives you

Everything in the single-app starter, available to every service:

- **Auth** — RS256 access and refresh tokens, Argon2 hashing, OAuth providers,
  and an enterprise IdP mode that validates someone else's tokens via JWKS.
- **LLM** — the OpenAI SDK directly, not LangChain. Chat completions and the
  Responses API, streaming, structured output, and error handling that maps
  provider failures to something your code can branch on.
- **Tool calling** — a tool is a function plus the Pydantic model describing
  its arguments; that model generates the schema the LLM sees and validates
  what comes back. `ToolRunner` executes the calls and loops until the model
  answers.
- **RAG** — chunk, embed, upsert, search, with the passage stored next to its
  vector. Re-ingesting replaces rather than accumulating stale chunks.
- **Usage** — every model call recorded at the service boundary, totalled per
  request, priced in `Decimal`. `UsageTracker().totals` answers "what did that
  cost".
- **Stores** — `AbstractStore[T]` and `AbstractSequenceStore`, one shared SQL
  implementation, and a conformance suite so a new backend is verified rather
  than assumed.
- **Infrastructure** — Redis or in-memory cache, Mongo, Cosmos, Azure Blob,
  Key Vault, Qdrant and Azure AI Search.

---

## How services talk

Not by reaching into each other's data. Through versioned events on a broker:

```python
await broker.apublish(Event.of(
    identity.USER_REGISTERED,
    {"user_id": str(user.id), "email": user.email},
    producer="identity",
))
```

`packages/contracts` holds the payload every side agrees on, versioned so a
producer can move without breaking a consumer. The broker is an abstraction
with an in-memory implementation for tests and Redis Streams for real use,
with consumer groups and at-least-once delivery — so handlers must be
idempotent, and dedupe on `event.meta.event_id`.

A correlation id is adopted from `X-Correlation-ID` or a W3C `traceparent`,
echoed on the response, and carried through the broker onto the worker, so one
logical request is traceable across the services it touches.

---

## Working on it

```bash
uv run ruff check .
uv run python tools/boundary_lint.py
uv run python tools/run_tests.py
```

Tests run **one service per process**. Configuration and SQL engines are
process-global by design, so running two services' suites in one interpreter
makes one reuse the other's database. That is what `run_tests.py` prevents —
use it rather than a bare `pytest` across the workspace.

`CONTRIBUTING.md` covers the rest.

---

## Tracking the single-app starter

This repository shares its foundation with
[opinionated-genai-starter](https://github.com/amrraedamr/opinionated-genai-starter).
A feature added there can be brought across without hand-porting:

```bash
uv run python tools/port_from_starter.py <sha> --check
```

It rewrites a commit's paths into the shared package or the owning service,
drops what has no destination here, and says what needs a human — rather than
silently putting a file in the wrong place.

---

## Design notes

`docs/architecture/` records why the structure is what it is: the library
boundary, data ownership, configuration composition, service contracts, build
and deploy, and the migration plan that got here.

---

## Licence

MIT. See [LICENSE](LICENSE).
