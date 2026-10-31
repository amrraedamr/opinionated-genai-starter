# Opinionated GenAI Starter

[![CI](https://github.com/amrraedamr/opinionated-genai-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/amrraedamr/opinionated-genai-starter/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-shaped FastAPI backend for LLM products, with the parts you would
otherwise spend three weeks rebuilding already wired together: auth, chat with
history, tool calling, RAG, and per-request cost tracking.

It is **opinionated** on purpose. There is one way to add a service, one way to
store an entity, one way to call a model. You can disagree with any of it, but
you will not have to decide it at 2am.

> Building several services rather than one app? There is a multi-service
> variant: [opinionated-genai-microservices-starter](https://github.com/amrraedamr/opinionated-genai-microservices-starter).

---

## Try it in five minutes

```bash
uv sync
mkdir -p keys database
openssl genrsa -out keys/private.pem 2048
openssl rsa -in keys/private.pem -pubout -out keys/public.pem
cp .env.example .env            # defaults run on SQLite
uv run alembic upgrade head
uv run api
```

Then open http://localhost:8000/docs.

`GETTING_STARTED.md` walks the same path with curl, from registering a user to
streaming a reply.

---

## What you get

**A real auth story.** Register and log in, RS256 access and refresh tokens,
Argon2 password hashing, Login-with-Google/Apple/etc through OAuth, and an
enterprise IdP mode that validates someone else's bearer tokens against their
JWKS. Routes protect themselves with a dependency, not an `if` statement.

**Chat that remembers.** Conversations and messages with ownership baked into
the query, so a user cannot read someone else's thread even if a handler
forgets to check. History is a store abstraction, so it can live in SQL or
Mongo without the agent noticing.

**Tool calling that runs.** Declare a tool as a function plus the Pydantic
model describing its arguments. The model generates the schema the LLM sees
and validates what comes back, so the two cannot drift. `ToolRunner` executes
the calls and feeds results back until the model answers.

```python
class Weather(BaseModel):
    city: str

async def get_weather(args: Weather) -> str:
    return f"4 degrees in {args.city}"

toolbox = Toolbox(Tool(
    name="get_weather",
    description="Current weather for a city.",
    parameters=Weather,
    handler=get_weather,
))
answer = await ToolRunner(toolbox).arun(messages)
```

**RAG without the glue.** Chunk, embed, upsert and search, with the passage
stored next to its vector so a result is text rather than an id you then have
to resolve. Re-ingesting a document replaces it instead of leaving stale
chunks behind to be retrieved later.

```python
await RagIngestionService(
    QdrantVectorSearchService(), IngestionConfig(collection="handbook")
).aingest([Document(id="leave-policy", text=policy)])

context = await RagRetrievalService(
    QdrantVectorSearchService(), "handbook"
).acontext("how do I request leave?")
```

**An answer to "what did that cost".** Every model call is recorded at the
service boundary, so no agent has to remember. Read the totals at the end of a
request.

```python
totals = UsageTracker().totals
totals.total_tokens, totals.cost_usd, totals.by_model
```

Cost is `Decimal`, and a model with no known rate is counted in tokens but
left out of the money rather than counted as free.

**Storage you can swap.** `AbstractStore[T]` for row-keyed entities,
`AbstractSequenceStore` for ordered items under a parent. One concrete SQL
implementation covers the CRUD, so a new store is its model and the columns
that are its own. Backends are held to a shared conformance suite, so "it
works with Mongo too" is a test rather than a claim.

**The rest of the plumbing.** Redis or in-memory cache, Azure Blob, Key Vault,
Qdrant / Azure AI Search / Mongo Atlas vectors, a prompt service with hot
reload, SSE streaming, structured error handling, and background tasks.

---

## How it is put together

```
Routers          thin controllers, auth via Depends
   ↓
Agents           one LLM turn, template-method shaped
   ↓
Services         LLM, prompts, cache, blob, vectors, usage
   ↓
Stores           entity CRUD, backend-agnostic
   ↓
Clients          SQLAlchemy, Redis, Mongo, Qdrant, Azure
```

Two ideas carry most of the weight.

**Pydantic models are canonical.** A view model is the shape of your data.
SQLAlchemy DTOs exist only to persist it, Mongo documents are
`model_dump()`, and the two never share a definition.

**Lifetimes are declared, not managed.** A metaclass says what something is:
`Singleton` for a plain service, `SingletonABC` when it implements an abstract
base, `KeyedSingleton` for one-per-config, `Stored` for one-per-request,
`NonInitialize` for a utility class that should never be instantiated. No
container, no wiring file.

```python
class EmailService(metaclass=Singleton): ...
class TokenCounter(metaclass=KeyedSingleton): ...
class ExecutionContext(metaclass=Stored): ...
```

---

## Configuration

One `Settings` class, read from the environment and `.env`, validated at
startup. Secrets can come from Azure Key Vault instead, and anything genuinely
secret should.

The defaults run on SQLite with local JWT keys, so a clone works before you
have configured anything. `.env.example` documents the rest, grouped by what
turns a feature on.

---

## Working on it

```bash
uv run pytest            # 122 tests, no infrastructure required
uv run ruff check .
uv run pre-commit install
```

Tests run against a throwaway SQLite database and deterministic fakes, so the
suite is green on a fresh clone without a single container.

Changed a model? Generate the migration rather than writing it:

```bash
uv run alembic revision --autogenerate -m "what changed"
```

`CONTRIBUTING.md` covers the conventions a review will ask about.

---

## Deployment

A single `Dockerfile` builds the app. `run.sh` picks Uvicorn locally and
Gunicorn with Uvicorn workers elsewhere, running migrations first when
`RUN_MIGRATIONS` is set.

```bash
docker build -t genai-api .
docker run -p 8000:8000 --env-file .env genai-api
```

---

## Outgrowing one app

When one deployable turns into several — separate release cadences, separate
databases, separate scaling — the structure here does not stretch to cover it,
and bending it is worse than moving.

[opinionated-genai-microservices-starter](https://github.com/amrraedamr/opinionated-genai-microservices-starter)
is the same foundation as a uv workspace: a shared `base` package plus
independently deployable services, each owning its own database, settings and
dependencies. `docs/going-multi-service.md` explains when that trade is worth
making, and features added here can be ported across with a tool rather than
by hand.

---

## Licence

MIT. See [LICENSE](LICENSE).
