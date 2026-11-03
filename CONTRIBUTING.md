# Contributing

Thanks for taking a look. This is a starter, so the bar for a change is a
little different from a product: it should make the thing people clone better,
not just add something to it.

## Getting set up

```bash
uv sync --all-packages
uv run pre-commit install
```

Each service has its own `.env.example`. Copy the one you're working on to
`.env` in that service's directory. The defaults run against SQLite, so a
service starts without any infrastructure. For the full stack:

```bash
docker compose -f deploy/docker-compose.infra.yml up -d
```

## Before you open a pull request

```bash
uv run ruff check .
uv run python tools/boundary_lint.py
```

CI runs the same two, then builds only the services your change affects. If
those pass locally they will pass there.

If you changed a dependency, commit the updated `uv.lock` alongside it.

## The boundaries that matter

`tools/boundary_lint.py` enforces these, so a violation fails CI rather than
review:

1. The shared `base` package imports no service.
2. No service imports another service.
3. `base` declares no table and no declarative root.
4. Each SQL service declares exactly one declarative root.
5. No foreign key crosses a service boundary.

A service owns its database. If you need another service's data, go through
its API or an event — never its tables. A reference to a row another service
owns is an opaque id with no foreign key behind it.

## Adding a service

```bash
uv run python tools/new_service.py --name billing --storage postgres --http
uv lock
```

That writes the whole skeleton: settings composed from the mixins it needs,
an app factory, Alembic wiring, and an `.env.example`.

## What tends to get asked in review

- **Don't repeat a helper.** If something already exists under
  `packages/base/base/tools`, use it. If you find yourself writing the same
  thing twice, it belongs there.
- **Settings compose.** `CoreSettings` holds only what every service shares.
  Anything else is a mixin a service opts into.
- **Type everything.** Full annotations, including on async counterparts.
- **Comment sparingly.** Explain a decision that isn't obvious from the code;
  don't narrate what the next line does.
- **Async names start with `a`** — `aget`, `aadd`, `acomplete` — rather than
  ending in `_async`.
- **Delete dead code** instead of commenting it out.

## Porting from the single-app starter

This repository tracks
[opinionated-genai-starter](https://github.com/amrraedamr/opinionated-genai-starter).
To bring a commit across:

```bash
uv run python tools/port_from_starter.py <sha> --check
```

It rewrites paths into the shared package or the owning service and tells you
which files need a human instead of guessing.

## Reporting something

Bugs and ideas are both welcome as issues. For a bug, the useful things are
what you ran, what you expected, and what happened instead. A failing test is
better than a description.
