# Getting Started

You have cloned a workspace with three services in it. This walks you from
that to a request crossing two of them and an event reaching a third, then
shows you how to add your own.

For what this is and why it is shaped this way, see [`README.md`](README.md).

---

## What you need

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** — for Postgres and Redis
- **openssl** — for the JWT key pair

---

## 1. Install the workspace

```bash
uv sync --all-packages
```

One `.venv` at the root holds every member. Do not `uv sync --package <one>`
on its own — it would uninstall the others. If you genuinely want a service in
isolation, give it its own environment:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/chat_api uv sync --package chat_api
```

## 2. Start the infrastructure

```bash
docker compose -f deploy/docker-compose.infra.yml up -d
```

Postgres (with `chat_api_db` and `identity_db` already created), Redis, Mongo
and Qdrant. Nothing else needs to be running yet.

## 3. Make the signing keys

`identity` signs tokens; everything else only verifies them. Both need a key,
and they must be the same pair:

```bash
mkdir -p keys
openssl genrsa -out keys/private.pem 2048
openssl rsa -in keys/private.pem -pubout -out keys/public.pem
```

## 4. Configure and migrate each service

Each service has its own `.env`, because each has its own database and its own
reason to exist.

```bash
for svc in identity chat_api; do
  cp services/$svc/.env.example services/$svc/.env
done
```

Then bring each schema up, from inside the service so its `.env` resolves:

```bash
cd services/identity && uv run alembic upgrade head && cd ../..
cd services/chat_api && uv run alembic upgrade head && cd ../..
```

Two Alembic histories, two databases. Neither knows about the other's tables.

## 5. Run them

Three terminals, or use the compose stack below.

```bash
cd services/identity   && uv run python -c "from identity.app import start; start()"
cd services/chat_api   && uv run python -c "from chat_api.app import start; start()"
cd services/ingestion_worker && uv run python -c "from ingestion_worker.worker import start; start()"
```

Or all of it, built and wired, in one command:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

---

## See the boundary work

Register and log in against **identity**:

```bash
curl -s -X POST http://localhost:8003/api/identity/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"Str0ng!Passw0rd","firstName":"You"}'

TOKEN=$(curl -s -X POST http://localhost:8003/api/identity/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"Str0ng!Passw0rd"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["accessToken"])')
```

Now use that token against **chat_api**, a different service with a different
database:

```bash
curl -s -X POST http://localhost:8001/api/7r/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"First"}'
```

It works because `chat_api` verifies the signature with the public key. It
never looks up a user, because it has no user table to look in:

```bash
docker exec -it genai-infra-postgres-1 psql -U local -d chat_api_db -c '\dt'
```

No `user`. That is the whole point.

And `chat_api` has no auth routes at all — token issuing belongs to identity:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8001/api/7r/auth/login
# 404
```

---

## See an event cross

`ingestion_worker` has no HTTP surface. It consumes what other services
publish, through a versioned contract rather than a shared table:

```python
from contracts.events import Event
from contracts.v1 import identity

await broker.apublish(Event.of(
    identity.USER_REGISTERED,
    {"user_id": str(user.id), "email": user.email},
    producer="identity",
))
```

The worker's log shows it arriving with the correlation id of the request that
caused it, so one logical operation stays traceable across three processes.

Redis Streams delivers at least once, so handlers must be idempotent — dedupe
on `event.meta.event_id`.

---

## Add your own service

```bash
uv run python tools/new_service.py --name billing --storage postgres --http
uv lock
```

That writes everything: settings composed from the mixins that storage choice
implies, an app factory, Alembic wiring, an `.env.example`. Then check you
have not broken a boundary:

```bash
uv run python tools/boundary_lint.py
```

If `billing` needs to know about a user, it holds the id and asks identity, or
listens for an event. It does not add a foreign key — the lint will tell you
so if you try.

---

## Where to go next

- [`README.md`](README.md) — the shape, the rules, and what the shared base
  gives you.
- [`docs/architecture/`](docs/architecture/) — why it is this way: the library
  boundary, data ownership, configuration composition, service contracts.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — the conventions a review will ask
  about, and the boundaries CI enforces.
- `tools/port_from_starter.py` — bringing a feature across from the single-app
  [opinionated-genai-starter](https://github.com/amrraedamr/opinionated-genai-starter).

## Reliability and production

For the real event path set `BROKER=redis` in all three services and give each service its own database. Apply migrations for `chat_api`, `identity`, and `ingestion_worker` before starting replicas. API writes commit an outbox row atomically; dispatchers publish after commit. The worker's durable inbox makes handlers idempotent.

`chat_api` must receive only `AUTH_PUBLIC_KEY_PATH`. Only `identity` receives `AUTH_PRIVATE_KEY_PATH`. Configure `TRUSTED_HOSTS`, `REQUEST_MAX_BYTES`, `REQUEST_CONCURRENCY_LIMIT`, `REQUEST_RATE_LIMIT_PER_MINUTE`, and `AUTH_RATE_LIMIT_PER_MINUTE`. These controls are intentionally process-local; enforce global limits and TLS at a gateway/ingress.

Docker Compose is a local developer stack, not a production topology. `docs/architecture/06-build-and-deploy.md` lists what a real deployment has to satisfy.
