"""Import every module in `base`, with every extra installed.

The other linters reason about declarations. This one actually runs the import
statements, which is the only way to catch a module that no service happens to
exercise: anything behind an extra no shipped service installs is never loaded
by the per-service jobs, so a stale import inside it reaches main unnoticed.

Run with all extras available, or the guarded modules will report as skipped
rather than passing.
"""

import importlib
import pathlib
import sys

from base.tools.errors import MissingExtraError

BASE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "packages/base"
PACKAGE: pathlib.Path = BASE / "base"


def module_names() -> list[str]:
    """Every importable module path under `base`, in a stable order."""
    names: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(BASE).with_suffix("")
        names.append(str(relative).replace("/", ".").removesuffix(".__init__"))
    return names


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

    print(f"Imports OK: {imported} module(s) in base load cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
