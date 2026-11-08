"""Enforce per-service dependency isolation.

Run from the workspace root:

    uv run python tools/deps_lint.py

An image carries its own service's dependency closure and nothing else.
`uv sync --package <service>` delivers that, but only while each service
declares exactly the `base` extras it uses, so both halves are checked here.

Rules, from docs/architecture/06-build-and-deploy.md:
  1. Every extra a service declares is one `base` actually defines.
  2. No service's production closure contains a distribution belonging only to
     an extra that service did not declare.
  3. `base`'s core dependencies match the pinned set below.

Rules 1 and 2 derive their deny list from `base`'s own manifest rather than a
hand-maintained one, so a new extra is covered the day it is added. Rule 3 is
deliberately a pinned list: a core dependency lands in every image ever built,
so growing that set should be a decision someone reviews, not a side effect.
"""

import re
import subprocess

import tomllib
from workspace import BASE, ROOT, SERVICES, service_names

BASE_EXTRAS = re.compile(r"^base\s*\[([^\]]*)\]")

CORE_DEPENDENCIES = {
    "aiohttp",
    "colorlog",
    "contracts",
    "httpx",
    "pydantic",
    "pydantic-settings",
    "pyhumps",
    "requests",
}
"""What `base` installs with no extras. Everything a shape-specific service
needs — an HTTP stack, a database driver, an LLM SDK — belongs to an extra.

`aiohttp` is core rather than an Azure extra: every async Azure client builds
a `DefaultAzureCredential`, which cannot import without it, and async SDKs
generally reach for it. Carrying it once beats discovering it missing per
extra."""


def normalize(requirement: str) -> str:
    """PEP 503 name of a requirement, without extras or version specifier."""
    name = re.split(r"[<>=!~\[;\s]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def extra_distributions() -> dict[str, set[str]]:
    """Distributions each `base` extra contributes directly."""
    optional = base_manifest()["project"].get("optional-dependencies", {})
    return {extra: {normalize(dep) for dep in deps} for extra, deps in optional.items()}


def base_manifest() -> dict:
    """`base`'s parsed manifest."""
    return tomllib.loads((BASE / "pyproject.toml").read_text())


def core_distributions() -> set[str]:
    """Distributions `base` installs with no extras selected."""
    return {normalize(dep) for dep in base_manifest()["project"]["dependencies"]}


def declared_extras(service: str) -> set[str]:
    """`base` extras the service asks for in its own dependencies."""
    manifest = tomllib.loads((SERVICES / service / "pyproject.toml").read_text())
    found: set[str] = set()
    for dependency in manifest["project"].get("dependencies", []):
        match = BASE_EXTRAS.match(dependency.strip())
        if match:
            found.update(
                part.strip() for part in match.group(1).split(",") if part.strip()
            )
    return found


def production_closure(service: str) -> set[str]:
    """Distributions the service's image installs, matching the Dockerfile's sync."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-hashes",
            "--no-emit-workspace",
            "--package",
            service,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return {
        normalize(line) for line in result.stdout.splitlines() if line[:1].isalnum()
    }


def main() -> int:
    extras = extra_distributions()
    failures: list[str] = []

    # 3. base's core stays the pinned set
    core = core_distributions()
    for dist in sorted(core - CORE_DEPENDENCIES):
        failures.append(
            f"base: {dist!r} is a core dependency, so every service image carries it. "
            f"Move it to an extra, or add it to CORE_DEPENDENCIES in this file."
        )
    for dist in sorted(CORE_DEPENDENCIES - core):
        failures.append(
            f"base: {dist!r} is pinned as core but is no longer declared. "
            f"Remove it from CORE_DEPENDENCIES in this file."
        )

    for service in service_names():
        declared = declared_extras(service)

        # 1. no extra that base does not define
        for extra in sorted(declared - extras.keys()):
            failures.append(
                f"{service}: declares base extra {extra!r}, which base does not define"
            )

        # 2. nothing belonging only to an extra this service did not declare
        allowed = {dist for extra in declared & extras.keys() for dist in extras[extra]}
        denied = {
            dist: extra
            for extra, dists in extras.items()
            if extra not in declared
            for dist in dists - allowed
        }
        for dist in sorted(denied.keys() & production_closure(service)):
            failures.append(
                f"{service}: ships {dist!r} from the {denied[dist]!r} extra, which it does not declare"
            )

    if failures:
        print("Dependency isolation violations:")
        for failure in sorted(set(failures)):
            print(f"  {failure}")
        return 1

    print(
        f"Dependency isolation OK: {len(service_names())} service(s) carry only their own closure"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
