# 0005 Experiment tracking: MLflow (sqlite) plus a narrative log

## Context

Training runs are the project's evidence. Earlier attempts left no record
of what was tried or why it failed, and the trained models themselves were
lost in a machine reorganisation.

## Decision

- Every training run is an MLflow run (sqlite store `mlflow.db`,
  artifacts under `mlruns/`, both gitignored) created through
  `scripts/explog.py`, which requires a `hypothesis` at start and records
  `outcome` and `diagnosis` at the end.
- `EXPERIMENTS.md` (tracked) mirrors each finished run and carries the
  narrative: baselines, ablation tables, milestones, what each failure
  turned out to be.
- Trained checkpoints live under `models/` (gitignored) and are published
  as release assets when they become a reference result.

## Consequences

- A new session reads `EXPERIMENTS.md` (or `python scripts/explog.py`)
  before proposing an attempt, so the same trap is not walked into twice.
- The sqlite store is local; the narrative file is the durable record.
