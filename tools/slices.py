"""Compose existing service slices into one service.

A *slice* is a reference service's domain code: its routers, ORM models, view
models and stores. `tools/new_service.py --from identity,chat_api` copies those
into a new service, rebinding them onto a single declarative root so one
deployment can serve auth and chat, while the same slices can still run as
separate services.

The reference services are the source. There is no second copy of the code to
drift from: `services/identity` *is* the auth slice, already exercised by CI,
`boundary_lint`, `deps_lint` and `preflight`.

Membership is by exclusion, not a path list. Every module in a service package
travels except the composition roots below, which the generator rewrites for
the target service. A file added to a reference service therefore joins its
slice automatically.
"""

import ast
import pathlib
import re
from dataclasses import dataclass, field

import tomllib
from workspace import SERVICES

# Regenerated for the target service rather than copied.
COMPOSITION_ROOTS: frozenset[str] = frozenset(
    {
        "app.py",
        "worker.py",
        "settings.py",
        "messaging.py",
        "__init__.py",
    }
)

# Present in more than one slice and merged rather than treated as a clash.
MERGEABLE: frozenset[str] = frozenset(
    {
        "models/orm/outbox.py",
        "models/definitions/database.py",
    }
)


class CompositionError(RuntimeError):
    """Raised when slices cannot be composed into one service."""


@dataclass(frozen=True)
class Capability:
    """A capability a slice contributes to `create_app`."""

    imports: tuple[str, ...]
    expr: str


@dataclass(frozen=True)
class Lifespan:
    """A lifespan hook a slice contributes to `create_app`."""

    import_line: str
    expr: str


@dataclass
class Slice:
    """One reference service, described well enough to copy into another."""

    name: str
    package: str
    directory: pathlib.Path
    summary: str
    mixins: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    lifespans: list[Lifespan] = field(default_factory=list)
    publishes_events: bool = False
    http: bool = True

    @property
    def package_dir(self) -> pathlib.Path:
        return self.directory / self.package

    @property
    def uses_sql(self) -> bool:
        return (self.directory / "alembic.ini").exists()

    @property
    def declarative_root(self) -> tuple[str, str] | None:
        """`(module, class_name)` of this slice's declarative root, if it has one."""
        for path in sorted(self.package_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = {
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                    for b in node.bases
                }
                if "DeclarativeBase" in bases:
                    rel = path.relative_to(self.package_dir).with_suffix("")
                    parts = list(rel.parts)
                    if parts[-1] == "__init__":
                        parts = parts[:-1]
                    module = ".".join([self.package, *parts])
                    return module, node.name
        return None

    def modules(self) -> list[pathlib.Path]:
        """Package-relative paths this slice contributes."""
        found: list[pathlib.Path] = []
        root_module = self.declarative_root
        root_path = None
        if root_module is not None:
            rel = root_module[0][len(self.package) + 1 :].replace(".", "/")
            root_path = f"{rel}/__init__.py" if rel else "__init__.py"
        for path in sorted(self.package_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(self.package_dir).as_posix()
            if rel in COMPOSITION_ROOTS or rel == root_path:
                continue
            # Namespace __init__.py files are recreated, not copied.
            if rel.endswith("__init__.py") and not path.read_text().strip():
                continue
            found.append(pathlib.Path(rel))
        return found


def load_slice(name: str) -> Slice:
    """Read a reference service's slice manifest."""
    directory = SERVICES / name
    manifest_path = directory / "pyproject.toml"
    if not manifest_path.exists():
        raise CompositionError(f"no service named {name!r} in services/")
    manifest = tomllib.loads(manifest_path.read_text())
    declared = manifest.get("tool", {}).get("starter", {}).get("slice")
    if declared is None:
        raise CompositionError(
            f"services/{name}/pyproject.toml declares no [tool.starter.slice], "
            f"so it cannot be composed. Add one, or generate a service without --from."
        )
    package = manifest["project"]["name"]
    return Slice(
        name=name,
        package=package,
        directory=directory,
        summary=declared.get("summary", ""),
        mixins=list(declared.get("mixins", [])),
        extras=list(declared.get("extras", [])),
        capabilities=[
            Capability(imports=tuple(c.get("imports", [])), expr=c["expr"])
            for c in declared.get("capabilities", [])
        ],
        lifespans=[
            Lifespan(import_line=lifespan["import"], expr=lifespan["expr"])
            for lifespan in declared.get("lifespans", [])
        ],
        publishes_events=bool(declared.get("publishes_events", False)),
        http=bool(declared.get("http", True)),
    )


def table_names(slice_: Slice) -> dict[str, str]:
    """`{table_name: module}` for every table this slice declares.

    Tables are named either by a string literal or by a member of the slice's
    own table enum (`__tablename__ = DatabaseTable.USER`), so enum members are
    resolved first and the assignment looked up against them.
    """
    enum_values: dict[str, str] = {}
    sources = {rel: (slice_.package_dir / rel).read_text() for rel in slice_.modules()}
    for text in sources.values():
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    enum_values[f"{node.name}.{item.targets[0].id}"] = item.value.value

    found: dict[str, str] = {}
    for rel, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__tablename__"
                    for t in node.targets
                )
            ):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found[value.value] = rel.as_posix()
            elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                resolved = enum_values.get(f"{value.value.id}.{value.attr}")
                if resolved:
                    found[resolved] = rel.as_posix()
    return found


def check_composable(slices: list[Slice]) -> None:
    """Fail loudly on clashes rather than emitting a service that breaks later."""
    if not slices:
        raise CompositionError("no slices given")

    non_http = [s.name for s in slices if not s.http]
    if non_http and len(slices) > 1:
        raise CompositionError(
            f"{', '.join(non_http)} has no HTTP surface and cannot yet be composed "
            f"with HTTP slices. Generate it as its own service instead."
        )

    seen_modules: dict[str, str] = {}
    for slice_ in slices:
        for rel in slice_.modules():
            key = rel.as_posix()
            if key in MERGEABLE:
                continue
            if key in seen_modules:
                raise CompositionError(
                    f"both {seen_modules[key]} and {slice_.name} provide {key}. "
                    f"Rename one before composing them."
                )
            seen_modules[key] = slice_.name

    seen_tables: dict[str, str] = {}
    for slice_ in slices:
        for table, module in table_names(slice_).items():
            if module in MERGEABLE:
                continue
            if table in seen_tables:
                raise CompositionError(
                    f"both {seen_tables[table]} and {slice_.name} declare a table "
                    f"named {table!r}. Rename one before composing them."
                )
            seen_tables[table] = slice_.name


def rewrite(text: str, slices: list[Slice], package: str, root: str) -> str:
    """Rebind a module onto the target package and declarative root."""
    for slice_ in slices:
        declarative = slice_.declarative_root
        if declarative is not None:
            source_module, source_class = declarative
            text = text.replace(
                f"from {source_module} import", f"from {package}.models.orm import"
            )
            text = re.sub(rf"\b{re.escape(source_class)}\b", root, text)
        text = re.sub(rf"\bfrom {re.escape(slice_.package)}\b", f"from {package}", text)
        text = re.sub(
            rf"\bimport {re.escape(slice_.package)}\b", f"import {package}", text
        )
    return text


def merge_database_definitions(slices: list[Slice], package: str) -> str | None:
    """Union every slice's table-name enum into one.

    Each slice names the tables it owns in a `StrEnum`. A composed service owns
    all of them, so the members are combined rather than one file winning.
    """
    members: list[tuple[str, str]] = []
    class_name = "DatabaseTable"
    for slice_ in slices:
        path = slice_.package_dir / "models" / "definitions" / "database.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            class_name = node.name
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value, ast.Constant)
                ):
                    entry = (item.targets[0].id, str(item.value.value))
                    if entry not in members:
                        members.append(entry)
    if not members:
        return None
    lines = [
        "from enum import StrEnum",
        "",
        "",
        f"class {class_name}(StrEnum):",
        f'    """Tables owned by {package}."""',
        "",
    ]
    lines += [f'    {name} = "{value}"' for name, value in members]
    return "\n".join(lines) + "\n"


def merge_outbox(slices: list[Slice], package: str, root: str) -> str | None:
    """Take one outbox table for the composed service.

    Every publishing slice ships the same `OutboxEventDTO`, differing only in
    its index name. A composed service dispatches from one outbox, so one copy
    travels and its index is renamed for the target.
    """
    for slice_ in slices:
        path = slice_.package_dir / "models" / "orm" / "outbox.py"
        if not path.exists():
            continue
        text = rewrite(path.read_text(), slices, package, root)
        return re.sub(
            r'"idx_\w*?outbox_pending"', f'"idx_{package}_outbox_pending"', text
        )
    return None
