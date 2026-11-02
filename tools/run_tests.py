"""Run every suite, each service in its own process.

    uv run python tools/run_tests.py

A service's settings and its SQL engine are process-global by design: one
process serves one service. Collecting two services into a single pytest run
makes the second reuse the first's engine, so each gets its own process here.
"""

import pathlib
import subprocess
import sys

from workspace import ROOT


def run(label: str, cwd: pathlib.Path) -> int:
    print(f"\n=== {label} ===", flush=True)
    return subprocess.run(["pytest", *sys.argv[1:]], cwd=cwd, check=False).returncode


def main() -> int:
    targets = [("base", ROOT)]
    services = ROOT / "services"
    if services.exists():
        targets += [
            (path.name, path)
            for path in sorted(services.iterdir())
            if (path / "tests").is_dir()
        ]

    failures = [label for label, cwd in targets if run(label, cwd) != 0]
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print(f"\nAll suites passed: {', '.join(label for label, _ in targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
