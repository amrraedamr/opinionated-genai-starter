# Contributing

Thanks for taking a look. This is a starter, so the bar for a change is a
little different from a product: it should make the thing people clone better,
not just add something to it.

## Getting set up

```bash
uv sync
uv run pre-commit install
```

Copy `.env.example` to `.env` and fill in what you need. The defaults run
against SQLite, so you can get the tests green without any infrastructure.

## Before you open a pull request

```bash
uv run ruff check .
uv run pytest
```

CI runs the same two, plus a lockfile check, a migration from an empty
database to head, and a package build. If those pass locally they will pass
there.

If you changed a dependency, commit the updated `uv.lock` alongside it.

If you changed an ORM model, generate the migration rather than writing it by
hand, and commit it:

```bash
uv run alembic revision --autogenerate -m "what changed"
```

Then run `alembic upgrade head` twice — the second run should be a no-op, and
autogenerating again should produce an empty migration. If it doesn't, the
model and the migration disagree.

## What tends to get asked in review

- **Tests.** New behaviour needs a test. A new `AbstractStore` implementation
  should subclass `AbstractStoreConformance` rather than hand-rolling its own
  coverage.
- **Don't repeat a helper.** If something already exists under `base/tools`,
  use it. If you find yourself writing the same thing twice, it belongs there.
- **Type everything.** Full annotations, including on async counterparts.
- **Comment sparingly.** Explain a decision that isn't obvious from the code;
  don't narrate what the next line does.
- **Async names start with `a`** — `aget`, `aadd`, `acomplete` — rather than
  ending in `_async`.
- **Delete dead code** instead of commenting it out.

## Reporting something

Bugs and ideas are both welcome as issues. For a bug, the useful things are
what you ran, what you expected, and what happened instead. A failing test is
better than a description.
