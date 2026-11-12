# 0004 Conventional commits, branch-per-task, protected master

## Context

Several agents and a human commit to this repository. History written as
"edit" commits (the repository's own past) is unusable for review or
bisection.

## Decision

- Every commit message follows Conventional Commits:
  `type(scope)?: subject`, type in `feat fix chore docs style refactor perf
  test build ci`, subject at most 72 characters. Enforced by the
  `commit-msg` hook and the `commits` CI job (merge and revert commits as
  git writes them are accepted).
- No commits on `master`. Branch names are `feat/…`, `fix/…`, `chore/…`,
  `docs/…`; one task per branch; merge through a pull request with the
  template filled in.
- `master` is protected on GitHub: pull requests required, the `check` and
  `commits` status checks required, force pushes and deletion disallowed.

## Consequences

- Any local auto-commit tooling that writes to `master` directly will be
  rejected at push time; it must commit on a branch.
- `git log` becomes a changelog; `EXPERIMENTS.md` stays the narrative log
  for experiments, which do not map to commits.
