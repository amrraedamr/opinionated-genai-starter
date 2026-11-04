"""Generate a new service from the conventions this workspace already uses.

    uv run python tools/new_service.py --name billing --storage postgres --http
    uv run python tools/new_service.py --name ingestion --storage mongo --no-http

Compose existing slices into one deployment with `--from`, so a single service
can serve auth and chat while the same slices still run separately:

    uv run python tools/new_service.py --name app --from identity,chat_api

Writes `services/<name>/` and nothing else. Run `uv lock` afterwards to add it
to the workspace lockfile.
"""

import argparse
import pathlib
import shutil
import subprocess
import textwrap

from slices import (
    CompositionError,
    Slice,
    check_composable,
    load_slice,
    merge_database_definitions,
    merge_outbox,
    rewrite,
)
from workspace import ROOT, SERVICES

SQL_STORAGE = {"postgres", "mysql", "sqlite"}

EXTRAS_FOR_STORAGE = {
    "postgres": ["sql", "postgres"],
    "mysql": ["sql", "mysql"],
    "sqlite": ["sql", "sqlite"],
    "mongo": ["mongo"],
    "none": [],
}


def write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    print(f"  {path.relative_to(ROOT)}")


def write_raw(path: pathlib.Path, content: str) -> None:
    """Write content that is already correctly indented."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  {path.relative_to(ROOT)}")


def declarative_root(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Base"


def compose(
    directory: pathlib.Path,
    pkg: str,
    name: str,
    slices: list[Slice],
    publishes_events: bool,
) -> list[str]:
    """Copy each slice's modules into the new service, rebound onto one root.

    Returns the ORM modules the generated Alembic environment must import so
    every table registers on the composed metadata.
    """
    root = declarative_root(name)
    package_dir = directory / pkg
    orm_modules: list[str] = []

    for slice_ in slices:
        for rel in slice_.modules():
            key = rel.as_posix()
            if key in ("models/definitions/database.py", "models/orm/outbox.py"):
                continue  # merged below, once, across every slice
            source = (slice_.package_dir / rel).read_text()
            write_raw(package_dir / rel, rewrite(source, slices, pkg, root))
            if key.startswith("models/orm/"):
                orm_modules.append(key[: -len(".py")].replace("/", "."))

    merged_tables = merge_database_definitions(slices, pkg)
    if merged_tables is not None:
        write_raw(package_dir / "models" / "definitions" / "database.py", merged_tables)

    if publishes_events:
        outbox = merge_outbox(slices, pkg, root)
        if outbox is not None:
            write_raw(package_dir / "models" / "orm" / "outbox.py", outbox)
            orm_modules.append("models.orm.outbox")
        write(
            package_dir / "messaging.py",
            f'''
            """One broker and one outbox dispatcher for every composed slice."""

            from collections.abc import AsyncGenerator
            from contextlib import asynccontextmanager

            from fastapi import FastAPI

            from base.services.broker import AbstractBroker
            from base.services.broker.factory import build_broker
            from base.services.messaging import OutboxDispatcher
            from {pkg}.models.orm.outbox import OutboxEventDTO
            from {pkg}.settings import settings

            BROKER: AbstractBroker = build_broker(settings)


            @asynccontextmanager
            async def messaging_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
                del app
                broker = BROKER
                dispatcher = OutboxDispatcher(broker, OutboxEventDTO)
                await dispatcher.astart()
                try:
                    yield
                finally:
                    await dispatcher.astop()
                    await broker.astop()
            ''',
        )

    # Namespace packages the copied modules live in.
    for parent in {rel.parent for s in slices for rel in s.modules()}:
        if parent == pathlib.Path("."):
            continue
        init = package_dir / parent / "__init__.py"
        if not init.exists():
            write(init, "")

    capability_imports: list[str] = []
    capability_exprs: list[str] = []
    lifespan_imports: list[str] = []
    lifespan_exprs: list[str] = []
    for slice_ in slices:
        for capability in slice_.capabilities:
            capability_imports += [i.format(pkg=pkg) for i in capability.imports]
            capability_exprs.append(capability.expr)
        for lifespan in slice_.lifespans:
            lifespan_imports.append(lifespan.import_line.format(pkg=pkg))
            lifespan_exprs.append(lifespan.expr)

    if any(s.uses_sql for s in slices):
        capability_imports.append("from base.capabilities.sql import SQLDependency")
        capability_exprs.append("SQLDependency()")
    if publishes_events:
        capability_imports.append("from base.capabilities.broker import BrokerDependency")
        capability_exprs.append("BrokerDependency(BROKER)")
        lifespan_imports.append(f"from {pkg}.messaging import BROKER, messaging_lifespan")
        lifespan_exprs.append("messaging_lifespan")

    imports = sorted(
        {f"from {pkg} import routers", *capability_imports, *lifespan_imports}
    )
    app_lines = [
        f'"""Composition root for {name}.',
        "",
        f"Composed from: {', '.join(s.name for s in slices)}.",
        "Regenerate rather than hand-editing if you re-compose.",
        '"""',
        "",
        "from base.services.configuration import configure",
        f"from {pkg}.settings import settings",
        "",
        "# Register the service's settings before importing anything that reads them.",
        "configure(settings)",
        "",
        "from base.app import create_app  # noqa: E402",
    ]
    app_lines += [f"{line}  # noqa: E402" for line in imports]
    app_lines += [
        "",
        "app = create_app(",
        "    settings=settings,",
        "    routers=routers,",
        "    prefix=settings.api_prefix,",
        "    capabilities=[",
    ]
    app_lines += [f"        {expr}," for expr in capability_exprs]
    app_lines += ["    ],"]
    if lifespan_exprs:
        app_lines += ["    lifespan_hooks=["]
        app_lines += [f"        {expr}," for expr in lifespan_exprs]
        app_lines += ["    ],"]
    app_lines += [
        ")",
        "",
        "",
        "def start() -> None:",
        "    from base.tools.webserver import run_webserver",
        "",
        f'    run_webserver(app_path="{pkg}.app:app", settings=settings)',
        "",
    ]
    write_raw(package_dir / "app.py", "\n".join(app_lines))
    return orm_modules


def build(
    name: str,
    storage: str,
    http: bool,
    capabilities: list[str],
    slices: list[Slice] | None = None,
) -> None:
    pkg = name
    directory = SERVICES / name
    if directory.exists():
        raise SystemExit(f"services/{name} already exists")

    slices = slices or []
    publishes_events = any(s.publishes_events for s in slices)
    if slices:
        http = any(s.http for s in slices)

    uses_sql = storage in SQL_STORAGE
    extras = ["server"] if http else []
    extras += EXTRAS_FOR_STORAGE[storage]
    extras += [c for c in capabilities if c not in extras]
    for slice_ in slices:
        extras += [e for e in slice_.extras if e not in extras]
    if publishes_events and "redis" not in extras:
        extras.append("redis")
    extras_str = ",".join(dict.fromkeys(extras))

    mixins = ["CoreSettings"]
    if http:
        mixins.append("ServerSettingsMixin")
    if uses_sql:
        mixins.append("SQLSettingsMixin")
        if storage == "postgres":
            mixins.append("PostgresSettingsMixin")
        elif storage == "mysql":
            mixins.append("MySQLSettingsMixin")
    if storage == "mongo":
        mixins.append("MongoSettingsMixin")
    if "redis" in capabilities or publishes_events:
        mixins.append("RedisSettingsMixin")
    if "llm" in capabilities:
        mixins.append("LLMSettingsMixin")
    if publishes_events:
        mixins.append("BrokerSettingsMixin")
    if slices:
        # A composed service verifies tokens; only the auth slice issues them.
        mixins.append("TokenVerificationMixin")
        for slice_ in slices:
            mixins += [m for m in slice_.mixins if m not in mixins]
    mixins = list(dict.fromkeys(mixins))
    # TokenIssuingMixin subclasses TokenVerificationMixin, so naming both would
    # be an unresolvable MRO. Issuing already provides verification.
    if "TokenIssuingMixin" in mixins and "TokenVerificationMixin" in mixins:
        mixins.remove("TokenVerificationMixin")
    mixin_imports = [m for m in mixins if m != "CoreSettings"]

    print(f"Creating services/{name}")

    manifest = textwrap.dedent(
        f"""
        [project]
        name = "{name}"
        version = "0.1.0"
        description = "{name} service"
        requires-python = ">=3.12,<3.15"
        dependencies = [
            "base[{extras_str}]",
        ]

        {'[project.scripts]' if http else ''}
        {f'serve = "{pkg}.app:start"' if http else ''}

        [tool.uv.sources]
        base = {{ workspace = true }}

        [tool.uv.build-backend]
        module-root = ""

        [build-system]
        requires = ["uv_build>=0.12.19,<0.13.0"]
        build-backend = "uv_build"
        """
    ).lstrip()

    # A SQL service whose backend is not SQLite still defaults to SQL_BACKEND=sqlite
    # locally, so it gets SQLite as a dev dependency. When storage *is* sqlite the
    # extra is already in `dependencies` and ships with the image.
    if uses_sql and storage != "sqlite":
        manifest += (
            "\n# This service runs its own backend in production and SQLite only\n"
            "# locally, so SQLite is a dev dependency. A service that runs SQLite\n"
            "# in production declares base[sqlite] above instead.\n"
            "[dependency-groups]\n"
            'dev = ["base[sqlite]"]\n'
        )

    write_raw(directory / "pyproject.toml", manifest)

    write(directory / pkg / "__init__.py", "")

    settings_lines = [
        f'"""{name} settings: `CoreSettings` plus the mixins this service uses."""',
        "",
        "from pydantic_settings import SettingsConfigDict",
        "",
        "from base.services.configuration.core import CoreSettings",
    ]
    if mixin_imports:
        settings_lines.append("from base.services.configuration.mixins import (")
        settings_lines += [f"    {m}," for m in sorted(mixin_imports)]
        settings_lines.append(")")
    settings_class = declarative_root(name).replace("Base", "") + "Settings"
    settings_lines += [
        "",
        "",
        f"class {settings_class}(",
    ]
    settings_lines += [f"    {m}," for m in mixins]
    settings_lines += [
        "):",
        "    model_config = SettingsConfigDict(",
        '        env_file=("../../.env.shared", ".env", ".env.local"),',
        '        env_file_encoding="utf-8",',
        '        extra="ignore",',
        "        populate_by_name=True,",
        "    )",
        "",
        f'    service_name: str = "{name}"',
        "",
        "",
        f"settings = {settings_class}()",
        "",
        "",
        'if __name__ == "__main__":  # pragma: no cover',
        "    # Validation entrypoint for CI and container preStart probes.",
        f'    print(f"{name} settings OK (environment={{settings.environment!r}})")',
        "",
    ]
    write_raw(directory / pkg / "settings.py", "\n".join(settings_lines))

    orm_modules: list[str] = []
    if slices:
        orm_modules = compose(directory, pkg, name, slices, publishes_events)
    elif http:
        write(directory / pkg / "routers" / "__init__.py", "")
        write(
            directory / pkg / "routers" / "health.py",
            f'''
            from fastapi import APIRouter

            router = APIRouter(prefix="/{name}", tags=["{name}"])


            @router.get("/ping")
            async def ping() -> dict[str, str]:
                return {{"service": "{name}"}}
            ''',
        )
        write(
            directory / pkg / "app.py",
            f'''
            """Composition root for {name}."""

            from base.services.configuration import configure
            from {pkg}.settings import settings

            # Register the service's settings before importing anything that reads them.
            configure(settings)

            from base.app import create_app  # noqa: E402
            from {pkg} import routers  # noqa: E402

            app = create_app(
                settings=settings,
                routers=routers,
                prefix=settings.api_prefix,
                capabilities=[],
            )


            def start() -> None:
                from base.tools.webserver import run_webserver

                run_webserver(app_path="{pkg}.app:app", settings=settings)
            ''',
        )

    if uses_sql:
        root = declarative_root(name)
        write(
            directory / pkg / "models" / "__init__.py",
            "",
        )
        write(
            directory / pkg / "models" / "orm" / "__init__.py",
            f'''
            """Declarative root for {name}."""

            from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

            from base.models.orm import SerializableMixin, TimestampMixin


            class {root}(MappedAsDataclass, DeclarativeBase, SerializableMixin):
                pass


            __all__ = ["{root}", "TimestampMixin"]
            ''',
        )
        write(
            directory / "alembic.ini",
            """
            [alembic]
            script_location = %(here)s/alembic
            prepend_sys_path = .
            path_separator = os

            [loggers]
            keys = root,sqlalchemy,alembic

            [handlers]
            keys = console

            [formatters]
            keys = generic

            [logger_root]
            level = WARNING
            handlers = console
            qualname =

            [logger_sqlalchemy]
            level = WARNING
            handlers =
            qualname = sqlalchemy.engine

            [logger_alembic]
            level = INFO
            handlers =
            qualname = alembic

            [handler_console]
            class = StreamHandler
            args = (sys.stderr,)
            level = NOTSET
            formatter = generic

            [formatter_generic]
            format = %(levelname)-5.5s [%(name)s] %(message)s
            datefmt = %H:%M:%S
            """,
        )
        write(directory / "alembic" / "versions" / ".gitkeep", "")
        env_imports = "\n".join(
            f"from {pkg}.{module} import *  # noqa: F401, F403" for module in sorted(orm_modules)
        )
        write_raw(
            directory / "alembic" / "env.py",
            "# Import each table so it registers on "
            f"{root}.metadata. Add a line per table.\n"
            "from base.tools.migrations import run_migrations\n"
            f"from {pkg}.models.orm import {root}\n"
            f"from {pkg}.settings import settings\n"
            + (env_imports + "\n" if env_imports else "")
            + f"\nrun_migrations({root}.metadata, settings)\n",
        )
        shutil.copyfile(
            SERVICES / "chat_api" / "alembic" / "script.py.mako",
            directory / "alembic" / "script.py.mako",
        )
        print(f"  services/{name}/alembic/script.py.mako")

    write(
        directory / ".env.example",
        f"""
        SERVICE_NAME={name}
        ENVIRONMENT=local
        LOG_LEVEL=INFO
        {'API_PREFIX=/api' if http else ''}
        """,
    )

    # Rewriting imports onto a new package reorders them, so leave the tree
    # lint-clean rather than handing over work the first CI run would reject.
    formatted = subprocess.run(
        ["ruff", "check", "--fix", "--quiet", str(directory)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if formatted.returncode not in (0, 1):
        print(f"  (ruff unavailable: {formatted.stderr.strip()})")

    print("\nNext:")
    print("  uv lock")
    print(f"  UV_PROJECT_ENVIRONMENT=.venvs/{name} uv sync --package {name}")
    print("  uv run python tools/boundary_lint.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="service name, snake_case")
    parser.add_argument(
        "--storage",
        choices=sorted(EXTRAS_FOR_STORAGE),
        default="none",
        help="backing store this service owns",
    )
    parser.add_argument("--http", action="store_true", default=True)
    parser.add_argument("--no-http", dest="http", action="store_false")
    parser.add_argument(
        "--capabilities",
        default="",
        help="comma-separated base extras, e.g. redis,llm",
    )
    parser.add_argument(
        "--from",
        dest="slices",
        default="",
        help="comma-separated services whose slices compose into this one, "
        "e.g. identity,chat_api",
    )
    args = parser.parse_args()

    if not args.name.isidentifier():
        raise SystemExit(f"{args.name!r} is not a valid python package name")

    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]

    slices: list[Slice] = []
    if args.slices:
        try:
            slices = [load_slice(s.strip()) for s in args.slices.split(",") if s.strip()]
            check_composable(slices)
        except CompositionError as exc:
            raise SystemExit(f"cannot compose: {exc}") from exc
        # A composed slice that owns tables needs a real backend; SQLite stays
        # the local default via the generated dev group.
        if args.storage == "none" and any(s.uses_sql for s in slices):
            args.storage = "postgres"

    build(args.name, args.storage, args.http, capabilities, slices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
