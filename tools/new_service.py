"""Generate a new service from the conventions this workspace already uses.

    uv run python tools/new_service.py --name billing --storage postgres --http
    uv run python tools/new_service.py --name ingestion --storage mongo --no-http

Writes `services/<name>/` and nothing else. Run `uv lock` afterwards to add it
to the workspace lockfile.
"""

import argparse
import pathlib
import shutil
import textwrap

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


def build(name: str, storage: str, http: bool, capabilities: list[str]) -> None:
    pkg = name
    directory = SERVICES / name
    if directory.exists():
        raise SystemExit(f"services/{name} already exists")

    uses_sql = storage in SQL_STORAGE
    extras = ["server"] if http else []
    extras += EXTRAS_FOR_STORAGE[storage]
    extras += [c for c in capabilities if c not in extras]
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
    if "redis" in capabilities:
        mixins.append("RedisSettingsMixin")
    if "llm" in capabilities:
        mixins.append("LLMSettingsMixin")
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

    if http:
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
        write(
            directory / "alembic" / "env.py",
            f'''
            # Import each table so it registers on {root}.metadata. Add a line per table.
            from base.tools.migrations import run_migrations
            from {pkg}.models.orm import {root}
            from {pkg}.settings import settings

            run_migrations({root}.metadata, settings)
            ''',
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
    args = parser.parse_args()

    if not args.name.isidentifier():
        raise SystemExit(f"{args.name!r} is not a valid python package name")

    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    build(args.name, args.storage, args.http, capabilities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
