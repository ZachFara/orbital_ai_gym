# 0002 Quality gates: ruff, mypy strict, vulture, import-linter, layout check

## Context

The code grew mission by mission across several agent sessions with no
automated checks. Future sessions need a single, non-negotiable bar, but
the existing 5,000 lines cannot be brought to strict typing and low
complexity in a setup phase without touching feature code.

## Decision

Tools, all configured in `pyproject.toml` and run by `make check`:

- **ruff format** as the formatter (line length 100) and **ruff check** as
  the linter with `E, F, W, I, B, UP, C90, SIM, N, RUF`. Four naming rules
  are off globally because physics code uses mathematical notation
  (`R`, `E`, `H`, `F`), and `E501` is off because the formatter owns line
  length.
- **Complexity**: McCabe limit 12 via ruff `C901`.
- **mypy --strict** for the whole tree.
- **vulture** at 80% confidence for dead code.
- **import-linter** contracts for the layer rules in DESIGN.md.
- **tools/check_layout.py** rejects tracked files outside the allowed
  directories and root-file list.
- **pytest** as the test runner.

**Baseline exceptions.** Files that failed the complexity or strict-typing
gates on the day the gates were added are listed explicitly in
`pyproject.toml`: `per-file-ignores` for `C901` (7 files), and a mypy
`overrides` block with `ignore_errors = true` for 17 legacy modules, which
are therefore *unchecked* until their burn-down task (an intermediate
"checked but untyped allowed" setting was tried and still left 190 errors
in annotated code against third-party types, i.e. real work, not setup).
One import-linter `ignore_imports` entry covers the single layer
violation. Modules that already pass strict (`presets`, the `__init__`
modules, `scripts/explog.py`, `tools/`, `tests/test_smoke.py`) are checked
in full from day one. These lists are a
**ratchet**: PROGRESS.md carries one burn-down task per entry, an entry is
deleted as soon as its file passes, and nothing is ever added to them
without a new decision record.

**Never loosen.** No new `noqa`, `type: ignore`, rule exclusions, raised
thresholds, skipped tests, or hook bypasses. A check that seems wrong goes
under Open Questions in PROGRESS.md.

## Consequences

- New code is held to the full bar from its first commit; old code is
  held to the bar it met on the day, and only gets stricter.
- `make check` is the only command anyone needs to remember.
- The baseline lists make the remaining debt visible in one place.
