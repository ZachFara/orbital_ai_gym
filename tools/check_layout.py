"""Reject tracked files outside the allowed layout.

Run by `make check`. Every tracked path must be either an allowed root file
or live under an allowed directory. The lists are the layout in DESIGN.md
and AGENTS.md; extend them there and here in the same change.
"""

from __future__ import annotations

import subprocess
import sys

ALLOWED_DIRS = (
    "orbital_gym/",
    "scripts/",
    "tests/",
    "tools/",
    "docs/",
    ".github/",
)

ALLOWED_ROOT_FILES = {
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "CLAUDE.md",
    "DESIGN.md",
    "EXPERIMENTS.md",
    "LICENSE",
    "Makefile",
    "PROGRESS.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    bad = [
        path
        for path in tracked_files()
        if path not in ALLOWED_ROOT_FILES and not path.startswith(ALLOWED_DIRS)
    ]
    if bad:
        print("files outside the allowed layout (see DESIGN.md / tools/check_layout.py):")
        for path in bad:
            print(f"  {path}")
        return 1
    print(f"layout ok ({len(tracked_files())} tracked files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
