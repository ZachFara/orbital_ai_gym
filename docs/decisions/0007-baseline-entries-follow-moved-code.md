# 0007 Baseline entries follow moved code

## Context

Decision 0002 forbids adding files to the quality-gate baseline lists.
Phase 2 moves legacy code to new module paths (the pilots from
`scripts/autopilot_demo.py` to `orbital_gym/autopilots/`), which would
otherwise force full strict typing and complexity work into a task whose
purpose is a move, and would be redone by the next task that rewrites the
same code.

## Decision

When a file on a baseline list is moved or split without being rewritten,
its entry moves with it (renamed to the new paths). The amount of unchecked
code does not grow; the burn-down task in PROGRESS.md is renamed to the new
path. A move that also rewrites the code must bring it to the gates instead.

## Consequences

- The ratchet holds: no new unchecked code, only relocated.
- Reviewers can see the relocation in the same diff as the move.
