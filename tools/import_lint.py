"""Import every module in `base`, with every extra installed.

The other linters reason about declarations. This one actually runs the import
statements, which is the only way to catch a module that no service happens to
exercise: anything behind an extra no shipped service installs is never loaded
by the per-service jobs, so a stale import inside it reaches main unnoticed.

It also resolves the mixin names each service declares for composition, since
those are strings in a manifest that no import statement would ever fail on.

Run with all extras available, or the guarded modules will report as skipped
rather than passing.
"""

import importlib
import pathlib
import sys

import tomllib

from base.tools.errors import MissingExtraError

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
BASE: pathlib.Path = ROOT / "packages/base"
PACKAGE: pathlib.Path = BASE / "base"
SERVICES: pathlib.Path = ROOT / "services"


def module_names() -> list[str]:
    """Every importable module path under `base`, in a stable order."""
    names: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(BASE).with_suffix("")
        names.append(str(relative).replace("/", ".").removesuffix(".__init__"))
    return names


def declared_mixins() -> dict[str, list[str]]:
    """`{service: mixin names}` from every `[tool.starter.slice]` manifest."""
    declared: dict[str, list[str]] = {}
    for manifest_path in sorted(SERVICES.glob("*/pyproject.toml")):
        manifest = tomllib.loads(manifest_path.read_text())
        slice_ = manifest.get("tool", {}).get("starter", {}).get("slice")
        if slice_ is None:
            continue
        declared[manifest_path.parent.name] = list(slice_.get("mixins", []))
    return declared


def unresolved_mixins() -> list[tuple[str, str]]:
    """Declared mixin names that no longer exist, as `(service, name)`.

    The scaffolder composes a new service from these strings, so a rename that
    misses a manifest produces a service that fails on its first import.
    """
    mixins = importlib.import_module("base.services.configuration.mixins")
    return [
        (service, name)
        for service, names in declared_mixins().items()
        for name in names
        if not hasattr(mixins, name)
    ]


def main() -> int:
    sys.path.insert(0, str(BASE))
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []
    imported = 0

    for name in module_names():
        try:
            importlib.import_module(name)
            imported += 1
        except MissingExtraError:
            # The extra is genuinely absent; the guard below reports it.
            skipped.append(name)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    if failures:
        print(f"Import check failed for {len(failures)} module(s):")
        for name, error in failures:
            print(f"  {name}: {error}")
        return 1

    if skipped:
        print(f"Import check incomplete: {len(skipped)} module(s) skipped.")
        print("Run with every extra installed: uv sync --all-extras")
        for name in skipped:
            print(f"  {name}")
        return 1

    unresolved = unresolved_mixins()
    if unresolved:
        print(f"Composition check failed for {len(unresolved)} declared mixin(s):")
        for service, name in unresolved:
            print(
                f"  services/{service}/pyproject.toml declares {name}, which no "
                f"longer exists in base.services.configuration.mixins"
            )
        return 1

    print(f"Imports OK: {imported} module(s) in base load cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
