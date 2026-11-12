# 0009 One benchmark protocol, one command

## Context

Each env had its own ad-hoc evaluation (the trainer's 32-site grid, the
fingerprint flights, hand-run demos). Publishing the envs needs numbers that
mean the same thing next month.

## Decision

- `orbital_gym.benchmark.protocol(env_id)` is the fixed episode set of an
  env: an even grid of landing sites where the env has one (8 for the
  missions, 16 for the practice ids, which run `nominal=True`), a run of
  seeds otherwise. Changing it is a decision, not a tweak.
- Three baselines, always: `random` (the action space sampled with the
  episode's seed), `autopilot` (the most capable scripted pilot for the
  mission; for a practice id, the task's expert started in its parking-orbit
  mode) and `ppo` (a model path per env, skipped when none is given).
- Metrics are the env's own: `info["success"]`, the summed reward, delta-v
  used, steps, and the histogram of terminations. No metric is defined by
  the benchmark itself.
- `python scripts/benchmark.py` prints the Markdown table that lives in the
  README; a result quoted anywhere else links back to that table and the
  commit it was made at. `--record DIR` writes the GIFs the README embeds.

## Consequences

- A pilot or env change that moves a row shows up as a README diff in the
  same PR.
- The practice ids' rows use the shaped reward, so their returns are not
  comparable with the missions' rows; the success column is.
