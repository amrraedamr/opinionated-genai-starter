"""Port a commit from the single-app starter into this workspace.

The starter keeps everything under `base/`. Here that split in two: shared
infrastructure under `packages/base/base/`, and the parts chat_api owns under
`services/chat_api/chat_api/`. This rewrites a starter patch's paths so it
applies cleanly.

    uv run python tools/port_from_starter.py <sha> [<sha> ...]
    uv run python tools/port_from_starter.py --range upstream/main~3..upstream/main

Paths with no mapping are reported, so a partial port is visible rather than
silent. Pass a stable ref or sha, not FETCH_HEAD: this fetches `upstream`
first and would overwrite it.
"""

import argparse
import re
import subprocess
import sys

BASE = "packages/base/base/"
CHAT_API = "services/chat_api/chat_api/"
IDENTITY = "services/identity/identity/"

# Paths the identity service owns.
IDENTITY_EXACT = {
    "models/orm/user.py",
    "models/orm/user_identity.py",
    "models/view_models/auth.py",
    "models/view_models/user.py",
    "routers/auth.py",
    "routers/user.py",
    "services/auth/__init__.py",
    "services/auth/dependencies.py",
    "services/auth/oauth/__init__.py",
    "services/store/user.py",
    "services/store/user_identity.py",
}

# Paths chat_api owns. Everything else under base/ is shared infrastructure.
CHAT_API_EXACT = {
    "models/definitions/database.py",
    "models/orm/chat_history_message.py",
    "models/orm/conversation.py",
    "models/orm/prompt.py",
    "models/view_models/chat.py",
    "models/view_models/conversation.py",
    "services/agents/test.py",
}
CHAT_API_PREFIXES = ("routers/", "services/prompt/templates/")

# Renamed during the split.
RENAMES = {"services/store/": "stores/"}

# Restructured beyond a path change; a patch touching these needs a human.
MANUAL = {
    "main.py": "split into packages/base/base/app.py and each service's app.py",
    "services/configuration/settings.py": "split into configuration/core.py and configuration/mixins.py",
    "services/auth/auths.py": "removed; the bearer scheme lives in base/services/auth/bearer.py",
}

# Paths outside `base/` that have no single destination here. A patch touching
# one needs a human, same as MANUAL, but the reason is about layout rather
# than a restructured module.
OUTSIDE = {
    "tests/": "the starter has one suite; here each service tests itself under services/<name>/tests/",
    "alembic/": "migrations are per service, under services/<name>/alembic/",
    "alembic.ini": "each SQL service has its own alembic.ini",
    "Dockerfile": "replaced by the parameterised docker/service.Dockerfile",
    "run.sh": "replaced by docker/run.sh",
    "pyproject.toml": "split into the workspace root plus a manifest per package and service",
}

DIFF_PATH = re.compile(r"^(diff --git |--- |\+\+\+ |rename from |rename to )(a/|b/)?(.+)$")


def map_path(path: str) -> tuple[str | None, str | None]:
    """Return (new_path, reason) where reason is set only when unmappable."""
    if not path.startswith("base/"):
        for prefix, reason in OUTSIDE.items():
            if path == prefix or path.startswith(prefix):
                return None, reason
        return path, None
    rel = path[len("base/") :]
    if rel in MANUAL:
        return None, MANUAL[rel]
    owner = IDENTITY if rel in IDENTITY_EXACT else None
    if owner is None and (rel in CHAT_API_EXACT or rel.startswith(CHAT_API_PREFIXES)):
        owner = CHAT_API
    if owner is None and not rel.startswith("services/store/"):
        return BASE + rel, None
    for old, new in RENAMES.items():
        if rel.startswith(old):
            return (owner or CHAT_API) + new + rel[len(old) :], None
    return (owner or CHAT_API) + rel, None


def rewrite(patch: str) -> tuple[str, list[str]]:
    """Rewrite every path in a starter patch.

    Returns the rewritten patch and a note per file that has no destination
    here. Those files are dropped rather than left to fail the whole apply —
    the rest of the commit still lands, and the note says what to do by hand.
    """
    header: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            sections.append((line, []))
        elif sections:
            sections[-1][1].append(line)
        else:
            header.append(line)

    out: list[str] = header
    problems: list[str] = []
    for first, body in sections:
        rewritten, why = _rewrite_section(first, body)
        if why:
            problems.append(why)
            continue
        out.extend(rewritten)
    return "".join(out), problems


def _rewrite_section(first: str, body: list[str]) -> tuple[list[str], str | None]:
    """Rewrite one file's diff, or explain why it has no destination here."""
    parts = first.rstrip("\n")[len("diff --git ") :].split(" b/")
    if len(parts) != 2:
        return [first, *body], None

    left = parts[0][2:] if parts[0].startswith("a/") else parts[0]
    new_left, why = map_path(left)
    if why:
        return [], f"{left}: {why}"
    new_right, why = map_path(parts[1])
    if why:
        return [], f"{parts[1]}: {why}"

    out = [f"diff --git a/{new_left} b/{new_right}\n"]
    for line in body:
        match = DIFF_PATH.match(line.rstrip("\n"))
        if not match:
            out.append(line)
            continue
        head, ab, rest = match.groups()
        if rest == "/dev/null":
            out.append(line)
            continue
        new, _ = map_path(rest)
        out.append(f"{head}{ab or ''}{new or rest}\n")
    return out, None


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shas", nargs="*", help="starter commits to port")
    parser.add_argument("--range", dest="rev_range", help="a commit range instead")
    parser.add_argument("--check", action="store_true", help="report only, apply nothing")
    parser.add_argument("--no-fetch", action="store_true", help="skip fetching upstream")
    args = parser.parse_args()

    if not args.shas and not args.rev_range:
        parser.error("give at least one sha, or --range")

    if not args.no_fetch:
        git("fetch", "upstream")
    shas = (
        git("rev-list", "--reverse", args.rev_range).split()
        if args.rev_range
        else args.shas
    )

    for sha in shas:
        subject = git("log", "-1", "--format=%s", sha).strip()
        print(f"\n=== {sha[:9]} {subject}")
        rewritten, problems = rewrite(git("format-patch", "-1", "--stdout", sha))

        for problem in problems:
            print(f"  MANUAL: {problem}")

        mode = ["--check"] if args.check else ["--index"]
        applied = subprocess.run(
            ["git", "apply", "--3way", *mode, "-"],
            input=rewritten,
            text=True,
            capture_output=True,
        )
        if applied.returncode != 0:
            print(f"  does not apply cleanly:\n{applied.stderr}")
            return 1
        if args.check:
            print("  applies cleanly")
            continue
        git("commit", "-m", subject, "-m", f"Ported from the starter ({sha[:9]}).")
        print("  applied and committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
