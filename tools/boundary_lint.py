"""Enforce the architectural boundaries the design depends on.

Run from the workspace root:

    uv run python tools/boundary_lint.py

Rules, each from docs/architecture:
  1. base imports no service.
  2. No service imports another service.
  3. base declares no table.
  4. base declares no DeclarativeBase.
  5. Each service declares exactly one declarative root.
  6. No ForeignKey crosses a service boundary.
"""

import ast

from workspace import BASE, ROOT, SERVICES, python_files, service_names


def imported_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def package_of(service: str) -> str:
    """Import package a service exposes, e.g. chat_api."""
    directory = SERVICES / service
    if (directory / service / "__init__.py").exists():
        return service
    for child in sorted(directory.iterdir()):
        if child.name == "alembic":
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            return child.name
    return service


def resolve_table(node: ast.AST, enum_values: dict[str, str]) -> str | None:
    """Resolve a table name from a constant, an enum member, or an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return enum_values.get(f"{node.value.id}.{node.attr}")
    if isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                resolved = resolve_table(part.value, enum_values)
                if resolved:
                    return resolved
            elif isinstance(part, ast.Constant) and isinstance(part.value, str):
                head = part.value.split(".")[0]
                if head:
                    return head
    return None


def foreign_key_table(node: ast.AST, enum_values: dict[str, str]) -> str | None:
    """Table a ForeignKey points at, from `"table.col"` or `f"{Enum.X}.col"`."""
    resolved = resolve_table(node, enum_values)
    return resolved.split(".")[0] if resolved else None


def main() -> int:
    services = service_names()
    packages = {name: package_of(name) for name in services}
    failures: list[str] = []

    # 1. base imports no service
    for path in python_files(BASE):
        imports = imported_modules(ast.parse(path.read_text()))
        for service, package in packages.items():
            if package in imports:
                failures.append(f"{path.relative_to(ROOT)}: base imports service package {package!r}")

    # 2. no service imports another service
    for service, package in packages.items():
        for path in python_files(SERVICES / service):
            imports = imported_modules(ast.parse(path.read_text()))
            for other, other_package in packages.items():
                if other != service and other_package in imports:
                    failures.append(
                        f"{path.relative_to(ROOT)}: {service} imports {other} package {other_package!r}"
                    )

    # 3 and 4. base declares no table and no declarative root
    for path in python_files(BASE):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__tablename__" for t in node.targets
            ):
                failures.append(f"{path.relative_to(ROOT)}: base declares __tablename__")
            if isinstance(node, ast.ClassDef):
                bases = {
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                    for b in node.bases
                }
                if "DeclarativeBase" in bases:
                    failures.append(
                        f"{path.relative_to(ROOT)}: base declares DeclarativeBase {node.name!r}"
                    )

    # 5. each service declares exactly one declarative root
    for service in services:
        roots: list[str] = []
        for path in python_files(SERVICES / service):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ClassDef):
                    bases = {
                        b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                        for b in node.bases
                    }
                    if "DeclarativeBase" in bases:
                        roots.append(f"{path.relative_to(ROOT)}:{node.name}")
        uses_sql = (SERVICES / service / "alembic.ini").exists()
        if uses_sql and len(roots) != 1:
            failures.append(
                f"{service}: expected exactly one DeclarativeBase, found {len(roots)} {roots}"
            )

    # 6. no ForeignKey crosses a service boundary
    for service in services:
        own_tables: set[str] = set()
        enum_values: dict[str, str] = {}
        for path in python_files(SERVICES / service):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__tablename__" for t in node.targets
                ):
                    if isinstance(node.value, ast.Constant):
                        own_tables.add(str(node.value.value))
                    else:
                        resolved = resolve_table(node.value, enum_values)
                        if resolved:
                            own_tables.add(resolved)
                # StrEnum members naming tables, e.g. USER = "user"
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if (
                            isinstance(item, ast.Assign)
                            and isinstance(item.value, ast.Constant)
                            and isinstance(item.value.value, str)
                            and len(item.targets) == 1
                            and isinstance(item.targets[0], ast.Name)
                        ):
                            enum_values[f"{node.name}.{item.targets[0].id}"] = item.value.value

        # Second pass, now that enum members are known.
        for path in python_files(SERVICES / service):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__tablename__" for t in node.targets
                ):
                    resolved = resolve_table(node.value, enum_values)
                    if resolved:
                        own_tables.add(resolved)

        for path in python_files(SERVICES / service):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", "")) == "ForeignKey"
                    and node.args
                ):
                    target = foreign_key_table(node.args[0], enum_values)
                    if target is None:
                        failures.append(
                            f"{path.relative_to(ROOT)}: ForeignKey target could not be resolved; "
                            f"boundary cannot be checked"
                        )
                    elif target not in own_tables:
                        failures.append(
                            f"{path.relative_to(ROOT)}: ForeignKey to {target!r}, "
                            f"not a table {service} owns"
                        )

    if failures:
        print("Boundary violations:")
        for failure in sorted(set(failures)):
            print(f"  {failure}")
        return 1

    print(f"Boundaries OK: base plus {len(services)} service(s) {services}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
