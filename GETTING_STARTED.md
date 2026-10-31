# Getting Started

A friendly entry point for this opinionated GenAI starter. If you've just cloned the repo and want to be up and running in a few minutes — and then know where to look next — you're in the right place.

For what this is and why it is shaped this way, see [`README.md`](README.md).

---

## What is this?

**opinionated-genai-starter** is a batteries-included FastAPI template for building GenAI applications. Instead of wiring auth, chat, agents, and infra from scratch every time, you fork or copy this starter and get a consistent, production-shaped foundation: layered architecture, metaclass-based DI, multi-database support, and pre-wired integrations for the usual GenAI stack.

It's aimed at:

- **Builders** who want a working chat / RAG / agent app in hours, not weeks.
- **Engineers** who want to swap one piece (say, the LLM provider, the SQL backend, or the vector DB) without rewiring the rest of the stack.
- **Reviewers** who want to read predictable code: routers stay thin, services own logic, stores own persistence, and abstractions only exist where multiple backends actually exist.

---

## What you get out of the box

- **Auth** — JWT (RS256) with access + refresh tokens, Argon2 password hashing, optional Login-with-X (Google/Facebook/Microsoft/GitHub/Apple), optional enterprise IdP Bearer validation, and an `Authorization: Bearer …` dependency you can drop on any route.
- **Conversations + chat history** — full conversation CRUD, canonical message storage with token counting, monotonic per-conversation sequencing, tool-call pair repair, visibility filtering.
- **Agents framework** — `AbstractAgent` template-method base; concrete agents read / write a per-request `ExecutionContext` and pipe streamed output through `StreamIterator`.
- **LLM via OpenAI SDK** — both the Chat Completions and the Responses API, sync + async, streaming + non-streaming, structured-output `parse_*` variants, automatic retries with jittered backoff and `Retry-After` support, Azure OpenAI content-filter detection on both pre-flight 400 responses and incomplete responses, and a typed `LLMError` hierarchy so callers never see raw SDK exceptions.
- **SSE streaming** — `StreamIterator` integrates with `sse-starlette`; the conversation router shows the full pattern (background-task pump + EventSourceResponse).
- **Vector search** — Qdrant, Azure AI Search, and MongoDB Atlas Search behind one `AbstractVectorSearchService` interface, plus idempotent Atlas Search index management.
- **Multi-database SQL** — SQLite (default), MySQL, or PostgreSQL behind one `SQLService` selected by config.
- **NoSQL** — MongoDB and Azure Cosmos DB behind one `AbstractNoSQLService`.
- **Cache** — Redis (with pub/sub + distributed locks) or in-memory for tests.
- **Azure** — Blob Storage, Cosmos DB, and `AzureKeyVaultService` for runtime vault access (settings themselves are env / `.env`; vault-as-settings-source is documented on `Settings` if you need it later).
- **Debug tracing** — every LLM call is captured (prompt, response, tokens, duration, errors) into a per-request `DebugTrace`; the conversation router persists it onto the assistant message metadata so you can replay any failure later.
- **Bilingual canned messages** — English / Arabic user-facing strings (`CannedMessages`) auto-mapped from the `LLMError` that triggered them, so end users see localized, safe messages instead of provider stack traces.
- **Live log streaming** — `POST /logs` returns an SSE stream of the application's stdout/stderr (authenticated).

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **OpenSSL** for generating the JWT key pair
- *(Optional)* **Docker** if you want to point at MySQL / Postgres / MongoDB / Redis containers instead of running with the bundled SQLite

---

## Setup in 5 steps

```bash
# 1. Install dependencies
uv sync

# 2. Generate the JWT RSA key pair
mkdir -p keys
openssl genrsa -out keys/private.pem 2048
openssl rsa -in keys/private.pem -pubout -out keys/public.pem

# 3. Configure
#    - Copy `.env.example` → `.env` and fill what you need. Defaults in Settings
#      already work for SQLite + local JWT keys.
#    - `mkdir -p database` (SQLite file path defaults to `database/app.db`).
#    - Set LLM keys (e.g. `OPENAI_LLM_API_KEY` or Azure OpenAI vars) when you
#      hit chat / embeddings endpoints.
#    - Optional Login-with-X: set OAUTH_PROVIDERS=google,apple,… plus each
#      provider’s client id/secret (Apple: TEAM_ID + KEY_ID + .p8). Optional
#      enterprise IdP Bearer: IDP_RESOURCE_SERVER_ENABLED + IDP_ISSUER/AUDIENCE/JWKS.
#      See `.env.example` for every variable, grouped by feature.

# 4. Apply database migrations
#    Creates the SQLite file (or updates the schema for MySQL / PostgreSQL).
uv run alembic upgrade head

# 5. Run the API
uv run api
# or, equivalently:
# uv run uvicorn base.main:app --reload
```

The API is now serving at `http://localhost:8000`.

---

## Open the docs

- **Swagger UI** — http://localhost:8000/docs
- **ReDoc** — http://localhost:8000/redoc
- **Health check** — http://localhost:8000/health-check

Swagger is the fastest way to explore the live API: every endpoint, schema, and authentication requirement is rendered automatically.

---

## Try it end-to-end (curl)

The following five commands take you from cold start to streaming a chat response. They assume the API is on `http://localhost:8000`. The auth endpoints accept JSON bodies with camelCase keys (Pydantic alias generation).

```bash
# 1. Register a user
curl -sS -X POST http://localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "demo@example.com",
    "password": "Demo!Pass123",
    "firstName": "Demo",
    "lastName": "User"
  }'

# 2. Log in and capture the access token
ACCESS_TOKEN=$(curl -sS -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email": "demo@example.com", "password": "Demo!Pass123"}' \
  | python -c 'import sys, json; print(json.load(sys.stdin)["data"]["accessToken"])')

# 3. Create a conversation
CONVERSATION_ID=$(curl -sS -X POST http://localhost:8000/api/conversations \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title": "Smoke test"}' \
  | python -c 'import sys, json; print(json.load(sys.stdin)["data"]["conversationId"])')

# 4. Send a non-streaming message
curl -sS -X POST "http://localhost:8000/api/conversations/$CONVERSATION_ID/messages" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"content": "Say hi in one short sentence."}'

# 5. List the messages so far
curl -sS "http://localhost:8000/api/conversations/$CONVERSATION_ID/messages" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

For SSE streaming, set `?streaming=true` and use `curl -N` so it doesn't buffer:

```bash
curl -N -X POST \
  "http://localhost:8000/api/conversations/$CONVERSATION_ID/messages?streaming=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"content": "Tell me a one-line joke about Python."}'
```

You'll see `StreamStart` → a sequence of `StreamMessage` events with token deltas → `StreamEnd`.

---

## Common tasks

### Add a new router

Drop a file in `base/routers/`, define a `router` named export, and the auto-discovery picks it up — no manual wiring.

```python
# base/routers/items.py
from fastapi import APIRouter

from base.models.orm.user import UserDTO
from base.services.auth.dependencies import validate_user_dependency

router = APIRouter(prefix="/items", tags=["Items"])


@router.get("")
async def list_items(current_user: UserDTO = validate_user_dependency) -> list[str]:
    return ["one", "two", "three"]
```

It's now mounted at `/api/items` after the next reload.

### Register a prompt

Drop a file in `base/services/prompt/templates/`, define a function returning a `Prompt`, and decorate it with `@register_prompt`. The `PromptLoader` auto-imports every module in that directory at startup.

A typical chat-aware template wires three items together: a system instruction, a `MessagesPlaceholder` for prior turns, and a structured user message:

```python
# base/services/prompt/templates/greeter.py
from base.models.view_models.prompt import (
    MessagesPlaceholder,
    Prompt,
    PromptTemplate,
    SystemMessageTemplate,
    UserMessageTemplate,
)
from base.services.prompt.decorator import register_prompt

GREETER_SYSTEM_PROMPT = """You are a friendly multilingual greeter. \
Reply in {language}."""

USER_INPUT_TEMPLATE = """{user_query}"""


@register_prompt
def greeter() -> Prompt:
    return Prompt(
        key="greeter",
        name="Greeter",
        description="Greet the user, optionally in context of prior turns.",
        template=PromptTemplate.from_messages([
            SystemMessageTemplate(content=GREETER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            UserMessageTemplate(content=USER_INPUT_TEMPLATE),
        ]),
    )
```

Render it with both the splice-in history and the structured variables:

```python
rendered = PromptService().render(
    "greeter",
    history=stored_messages,  # list[StoredMessage], or [] if you don't have any
    user_query="Hello!",
    language="Arabic",
)
```

`MessagesPlaceholder` defaults to `optional=True`, so passing `history=[]` (or a missing `history` key) is fine. It rejects `system` / `developer` roles inside the placeholder — those are instructions, not conversation.

### Write an agent

Subclass `AbstractAgent`, set `PROMPT_KEY`, and implement `_execute()`. `arun()` handles the LLM error → canned message → `FlowAbortError` mapping for you. Use the built-in `_aget_history()` helper to feed prior turns into the prompt's `MessagesPlaceholder` — it slices off the trailing `USER` message (which is the current query, already substituted via `{user_query}`):

```python
# base/services/agents/greeter.py
from typing import ClassVar

from base.services.agents import AbstractAgent
from base.services.llm import OpenAIMessageAdapter


class GreeterAgent(AbstractAgent):
    PROMPT_KEY: ClassVar[str] = "greeter"
    MAX_OUTPUT_TOKENS: ClassVar[int] = 200
    HISTORY_LIMIT: ClassVar[int | None] = 20

    async def _execute(self) -> None:
        ctx = self._execution_context
        rendered = self._prompt_service.render(
            self.PROMPT_KEY,
            history=await self._aget_history(),
            user_query=ctx.flow_variables.user_query,
            language=ctx.flow_variables.locale.value,
        )
        result = await self._aresponses(
            OpenAIMessageAdapter.to_responses_input(prompt=rendered),
        )
        ctx.flow_variables.final_answer = (result.content or "").strip()
```

Call it from a router with `await GreeterAgent().arun()` after populating the `ExecutionContext`.

---

## Where to go next

- [`README.md`](README.md) — what this is, what you get, and how it fits together.
- **http://localhost:8000/docs** — the endpoint matrix, generated from the code
  rather than written down and left to rot.
- [`.env.example`](.env.example) — every setting, grouped by the feature it
  turns on: OAuth providers, enterprise IdP, LLM keys, storage backends.
- [`base/services/agents/__init__.py`](base/services/agents/__init__.py) — the
  agent template method, and the hooks a concrete agent fills in.
- [`base/services/tool_calling/`](base/services/tool_calling/) — declaring a
  tool and letting the model call it.
- [`base/services/rag/`](base/services/rag/) — chunking, ingestion, retrieval.
- [`base/services/llm/openai_service.py`](base/services/llm/openai_service.py)
  — both OpenAI surfaces, the `LLMError` hierarchy, and retry behaviour.
- [`docs/going-multi-service.md`](docs/going-multi-service.md) — when one app
  should become several, and what that costs.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — the conventions a review will ask about.
