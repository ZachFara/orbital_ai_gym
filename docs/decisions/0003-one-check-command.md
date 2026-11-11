# 0003 One check command, wired identically into pre-commit and CI

## Context

Checks that differ between a developer's machine and CI drift apart, and
agents learn to run the subset that passes locally.

## Decision

- `make check` is the one check. It runs the formatter (check mode), the
  linter, the type checker, dead-code detection, the import contracts, the
  layout check and the test suite, in that order, and stops at the first
  failure.
- The pre-commit hook (`.pre-commit-config.yaml`, installed by `make
  hooks`) is a local hook whose entry is literally `make check`.
- CI (`.github/workflows/check.yml`, job `check`) installs the project
  with every extra and runs `make check`.
- A second job, `commits`, runs `tools/check_commit_msg.py` over every
  commit in a pull request; the same script is the `commit-msg` hook.
- `make fmt` applies the formatter and ruff's safe fixes; it is the only
  write-mode tool invocation.

## Consequences

- There is nothing to configure twice. Changing a check means editing
  `pyproject.toml` or the Makefile, and both hook and CI pick it up.
- A commit takes a few seconds longer (the suite runs in ~5 s today).
  If the suite grows past what a commit can bear, split the Makefile
  target into `check-fast` and `check`, keep CI on `check`, and record
  the decision.
