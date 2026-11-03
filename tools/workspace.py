"""Workspace layout, shared by the scripts in this folder.

These run as plain scripts (`uv run python tools/<name>.py`), so `tools/` is on
`sys.path` and this module is importable as `workspace`.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = ROOT / "packages"
SERVICES = ROOT / "services"
BASE = PACKAGES / "base"


def service_names() -> list[str]:
    """Every service directory in the workspace, sorted."""
    if not SERVICES.exists():
        return []
    return sorted(p.name for p in SERVICES.iterdir() if p.is_dir())


def python_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Source files under `root`, skipping caches, virtualenvs and migrations."""
    return [
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
        and ".venv" not in p.parts
        and "alembic" not in p.parts
    ]
