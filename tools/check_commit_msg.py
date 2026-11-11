"""Conventional-commit message check.

    python tools/check_commit_msg.py <message-file>      # commit-msg hook
    python tools/check_commit_msg.py --range A..B        # CI, every commit in range

Accepted: `type(scope)!: subject` with type in TYPES, subject <= 72 chars.
Merge commits and reverts are accepted as git writes them.
"""

from __future__ import annotations

import re
import subprocess
import sys

TYPES = ("feat", "fix", "chore", "docs", "style", "refactor", "perf", "test", "build", "ci")
PATTERN = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9_./-]+\))?!?: \S.{{0,71}}$")
PASSTHROUGH = re.compile(r"^(Merge |Revert )")


def valid(subject: str) -> bool:
    return bool(PATTERN.match(subject) or PASSTHROUGH.match(subject))


def check_range(rev_range: str) -> int:
    out = subprocess.run(
        ["git", "log", "--format=%H %s", rev_range], check=True, capture_output=True, text=True
    )
    bad = [line for line in out.stdout.splitlines() if not valid(line.split(" ", 1)[1])]
    for line in bad:
        print(f"not a conventional commit: {line}")
    return 1 if bad else 0


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if not ln.startswith("#")]
    subject = lines[0].strip() if lines else ""
    if valid(subject):
        return 0
    print(
        f"not a conventional commit message: {subject!r}\n"
        f"expected: <type>(<scope>)?: <subject>  with type in {', '.join(TYPES)}"
    )
    return 1


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--range":
        return check_range(argv[2])
    if len(argv) == 2:
        return check_file(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
