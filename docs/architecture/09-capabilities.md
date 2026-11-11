# 09 — Capabilities

Six capabilities: tenancy, tracing, rate limiting, guardrails, MCP and realtime. Rate
limiting is part of every HTTP service, because the traffic middleware that spends it
is. The other five are off until a service asks, so a service that asks for none pays
for none — no column, no dependency, no extra work per request.

The pattern is the same for all six. A **settings mixin** says a service wants it,
an **extra** installs what it needs, and the code reads the mixin's **protocol**, so a
service that never composed the mixin gets a clear `MissingSettingsError` rather than
an `AttributeError` somewhere deep.

| Capability | Mixin | Extra | Off unless |
| --- | --- | --- | --- |
| Tenancy | `TenancySettingsMixin` | — | `TENANCY_ENABLED=true` |
| Tracing | `TelemetrySettingsMixin` | `otel` | `OTEL_ENABLED=true` |
| Rate limiting | `RateLimitSettingsMixin` | `redis` for the Redis backend | always on for HTTP services |
| Guardrails | `GuardrailSettingsMixin` | `openai` for moderation | `GUARDRAILS_ENABLED=true` |
| MCP | `McpClientSettingsMixin`, `McpServerSettingsMixin` | `mcp` | servers configured / `MCP_SERVER_ENABLED=true` |
| Realtime | `RealtimeSettingsMixin` | `openai` or `gemini` | a route uses it |

---

## Tenancy

Which customer a request acts for, and that nobody acts for one unasked.

The tenant comes from a token claim, `tid` by default (`TENANT_CLAIM`), the one Entra
ID issues. It is read **only** where a service set `TENANCY_ENABLED=true`: an identity
provider sends `tid` on every token, and a service that read it without asking would
start partitioning data it never meant to.

`tenant_id_dependency` gives a route the caller's tenant and refuses a token naming
none, so a scoped query never runs for nobody. Below the route, `TenantContext` holds
the tenant for the rest of the request, and it is carried across the broker: the
outbox stamps each event with it, and the consumer runs inside it.

### Scoping stores

A model opts in by carrying `TenantScopedMixin`. `SQLStore` then narrows every read to
the tenant in context, stamps every write with it, and refuses an `edit` that would move
a row into another tenant. With no tenant in context it raises rather than reading
`tenant_id IS NULL`.

### Row-level security, the second fence

Store scoping is one layer; a query that bypasses the store bypasses it too. Postgres
row-level security is an optional second one, enabled with `TENANCY_RLS_ENABLED=true`.
Each transaction is then told its tenant through `set_config('app.tenant_id', …)`, and
a migration fences each table with the policy `RowLevelSecurity.statements(table)`
emits (`drop_statements` reverses it):

```python
from base.services.sql.rls import RowLevelSecurity

def upgrade() -> None:
    for statement in RowLevelSecurity.statements("conversation"):
        op.execute(statement)
```

pgvector collections get a tenant column and the same policy when RLS is on. **A
superuser or a role with `BYPASSRLS` ignores every policy**, so connect as an ordinary
role — the local compose superuser is exempt.

---

## Tracing

OpenTelemetry over OTLP, built per worker in `astartup` so processes forked by a
preloading server do not share an exporter. `OTEL_ENABLED=true` turns it on; the
endpoint, service name and sample ratio are settings beside it.

Spans leave through a **redacting exporter**: attributes, events, exception messages,
the status description and string lists all pass through PII redaction before export,
and dropped counts are kept. Redaction is the guardrails' PII check, so it applies once
a service turns guardrails and PII redaction on (below). Prompts and replies are not recorded at all unless
`OTEL_CAPTURE_CONTENT=true`.

Locally, services export to the collector in `deploy/docker-compose.infra.yml`, which
forwards to Jaeger at http://localhost:16686. The collector is in the path because a
real deployment has one to batch, sample and strip attributes.

---

## Rate limiting

Five budgets, each per minute:

| Budget | Setting | Spent by | Counted per |
| --- | --- | --- | --- |
| Requests | `RATE_LIMIT_REQUESTS_PER_MINUTE` | every HTTP request | client address |
| Auth | `RATE_LIMIT_AUTH_PER_MINUTE` | paths under `/auth/` | client address |
| Socket connects | `RATE_LIMIT_WS_CONNECT_PER_MINUTE` | opening a websocket | client address |
| LLM tokens | `RATE_LIMIT_LLM_TOKENS_PER_MINUTE` | model calls, by tokens used | user within tenant |
| Socket messages | `RATE_LIMIT_WS_MESSAGES_PER_MINUTE` | each websocket frame | user within tenant |

The first three are spent in middleware, before authentication has run, so the client
address is all that is known; behind a proxy, every client shares the proxy's address.
The last two are spent where the caller is known, and keyed by **user within their
tenant**, falling back to the tenant, the user, then the address. Keying by tenant alone
would let one heavy user exhaust everyone's.

`RATE_LIMIT_BACKEND` chooses where counters live. `memory` counts per replica. `redis`
counts across the fleet with GCRA on Redis's own clock, so replicas with skewed clocks
agree. When Redis fails, the Redis limiter falls back to an in-memory floor rather than
to no limit, and stops asking Redis for a few seconds so a hung Redis does not add its
timeout to every request.

The LLM token budget is a check before a call and a charge after it. The charge records
what was used even past the limit, so the next call is the one refused.

---

## Guardrails

Checks on what goes into a model, what comes out, and what tools hand back.
`GUARDRAILS_ENABLED=true` builds the pipeline; each guardrail has its own switch.

| Guardrail | Stages | Does |
| --- | --- | --- |
| PII | every stage | Redacts email, card, IBAN and phone numbers, plus `PII_EXTRA_PATTERNS` |
| Moderation | input, output | Refuses content OpenAI's moderation endpoint scores at or above `MODERATION_THRESHOLD` |
| Injection | tool results | Refuses instruction-override text in what a tool returned |

The **tool result** stage is the reason third-party tools are safe to enable: an MCP
server nobody here reviewed returns text the model was told to trust, so it is checked on
the way in, not only on the way back out.

### Streaming

A streamed reply is checked as it streams. Cheap local checks such as PII run on each
fragment (`OUTPUT_DELTA`); moderation, which bills per call and cannot judge half a
sentence, sees only the complete reply. The stream is cut only where checking the two
halves gives the same answer as checking the whole, so a card number split across
chunks is still caught. A reply refused as a whole ends in a retraction rather than a
normal end. `PII_STREAM_TAIL` is how much is held back; 48 covers the longest spaced
IBAN.

### History

Earlier turns are saved raw, so they are re-checked each turn: redacted, or replaced
with a placeholder when blocked. Only the newest turn can refuse a request, so one
blocked message does not block the conversation forever. Reports are cached by content
hash, so moderation is not billed again for the same history.

### Logs and traces

Credentials and bearer tokens are stripped from every log line in every service,
unconditionally. Personal data is the guardrails' decision: with `GUARDRAILS_ENABLED=true`
and `PII_REDACTION_ENABLED=true`, the same PII redaction runs on every log line and every
exported span, and a service that does not compose `GuardrailSettingsMixin` leaves it in.
A pattern that does not compile is refused at startup, not on the first log line.

---

## MCP

The Model Context Protocol, both ways: a service can call tools on other MCP servers,
and can offer its own toolbox to any MCP client. Both live behind the `mcp` extra.

### Calling other servers

`McpClients` connects to every server in `MCP_SERVERS` at startup and offers their tools
as one `Toolbox`:

```bash
MCP_SERVERS='[{"name": "docs", "transport": "http", "url": "https://docs.internal/mcp/", "credential": "docs"}]'
```

```python
mcp = McpClients(credentials={"docs": OutboundCredentials().provider})
app = create_app(settings=settings, capabilities=[mcp, ...])

# inside a request, after startup
ToolRunner(mcp.toolbox).arun(messages)
```

Each tool is named `server__tool`, so two servers offering `search` do not collide. A
name no model provider accepts — MCP allows dots and 128 characters, providers allow
neither — is refused at startup rather than at the provider. A server's token comes
from a `TokenProvider` named by `credential`, so no secret sits in the server list.

Sessions are held for the service's lifetime and reconnect once when a server drops;
`/readyz` fails while a server is unreachable. Transports are streamable HTTP, and stdio
for a server run as a subprocess (`"transport": "stdio", "command": …, "args": […]`).

### Offering a toolbox

`McpServer` mounts a toolbox at `MCP_SERVER_PATH` (`/mcp`) when `MCP_SERVER_ENABLED=true`:

```python
app = create_app(settings=settings, capabilities=[McpServer(toolbox)])
```

Tools are listed and called straight from the toolbox, so a tool's schema and validation
are the ones every agent already uses, whether it was described by a Pydantic model or
by raw JSON Schema. Every call needs a bearer this service can verify, and runs as that
caller's tenant and user, as a route would. A tool that raises is reported to the calling
model as a failure; the exception text stays in this service's log.

`services/mcp_server` is a reference service built on it.

---

## Realtime

A held-open conversation with a model over a websocket, spoken or typed, alongside
per-user delivery that reaches every device a user has.

### Connecting

A browser's `WebSocket` cannot set an `Authorization` header, so a socket authenticates
with a **ticket**: `POST /realtime/tickets` with a bearer returns a single-use ticket
valid for 30 seconds, and the client opens the socket passing it in
`Sec-WebSocket-Protocol` as `ticket.<value>`. Not in the query string, which lands in
access logs. Send back the `device_id` from an earlier ticket so a reconnect resumes.

Each socket has a 64 KB frame limit, a per-message rate limit, and a heartbeat that
closes a peer silent for 60 seconds.

### Talking to a model

`/realtime/conversation` bridges the socket to a provider session — OpenAI's realtime
API or Gemini Live, chosen by `REALTIME_PROVIDER`. Each direction has its own pump, so a
user can talk over the model and a tool can run mid-sentence without either waiting.

Text is guarded in both directions, and the transcript of what the model heard is
checked as the user's own words; audio itself is samples, not sentences, and is not.
Tools come from the same `Toolbox` agents use. Each turn is checked against the LLM
token budget before input is forwarded and charged when it ends; an exhausted budget
tells the client and keeps the socket open.

### Delivery to every device

`ConnectionManager` reaches a user wherever they are connected, across replicas, over
Redis pub/sub. A channel is one of two kinds:

- **Ephemeral** — reaches whoever is connected now.
- **Durable** — written to a per-user inbox first, with a sequence number. A reconnecting
  device sends `resume` with its last sequence and receives the gap; `ack` advances its
  cursor. One inbox, one cursor per device. The inbox keeps its newest 1,000 entries,
  and is dropped a day after its last write.

Presence is tracked per socket, so a user is online while any replica holds one of their
devices, and keeps a last-seen time for a week after the last one closes. On shutdown a
replica tells its clients to reconnect elsewhere before closing them.
