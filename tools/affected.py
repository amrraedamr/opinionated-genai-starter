"""Which services a change affects, for a CI build matrix.

    uv run python tools/affected.py                      # vs origin/main
    uv run python tools/affected.py --base HEAD~1
    uv run python tools/affected.py --format table

Prints a JSON array by default, suitable for a GitHub Actions matrix.
"""

import argparse
import json
import subprocess
import sys

import tomllib
from workspace import SERVICES, service_names

# A change to any of these rebuilds every service.
GLOBAL_PATHS = ("uv.lock", "pyproject.toml", "docker/", "deploy/", ".github/")

# A change to these affects nothing buildable.
IGNORED_PATHS = ("docs/", "README.md", "GETTING_STARTED.md", ".gitignore")


def workspace_deps(service: str) -> set[str]:
    """Workspace packages a service declares, from its pyproject."""
    manifest = SERVICES / service / "pyproject.toml"
    if not manifest.exists():
        return set()
    data = tomllib.loads(manifest.read_text())
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    declared = {
        name
        for name, spec in sources.items()
        if isinstance(spec, dict) and spec.get("workspace")
    }
    return declared


def changed_files(base: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"], capture_output=True, text=True, check=False
    )
    ref = merge_base.stdout.strip() or base
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git diff failed: {result.stderr}", file=sys.stderr)
        raise SystemExit(2)
    return [line for line in result.stdout.splitlines() if line]


def affected(paths: list[str]) -> list[str]:
    all_services = service_names()
    if not paths:
        return []

    hit: set[str] = set()
    for path in paths:
        if path.startswith(IGNORED_PATHS):
            continue
        if path.startswith(GLOBAL_PATHS) or path == "uv.lock":
            return all_services
        if path.startswith("services/"):
            parts = path.split("/")
            if len(parts) > 1 and parts[1] in all_services:
                hit.add(parts[1])
            continue
        if path.startswith("packages/"):
            parts = path.split("/")
            if len(parts) > 1:
                package = parts[1]
                for service in all_services:
                    if package in workspace_deps(service):
                        hit.add(service)
            continue
        if path.startswith("tools/"):
            continue
    return sorted(hit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="ref to compare against")
    parser.add_argument("--format", choices=("json", "table"), default="json")
    args = parser.parse_args()

    paths = changed_files(args.base)
    result = affected(paths)

    if args.format == "table":
        print(f"changed files: {len(paths)}")
        for path in paths[:40]:
            print(f"  {path}")
        print(f"\naffected services: {result or 'none'}")
    else:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
