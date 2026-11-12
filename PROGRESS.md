# PROGRESS

## Status (2026-11-12)

Phase 1 merged (PR #1); repository public with branch protection. Phase 2
in progress: A1-A6 done. Four missions exist and are solved by scripted pilots;
a learned PPO policy lands the moon mission on the site 48/48 from the
launch pad (`EXPERIMENTS.md`). `ToMars-v0` has no RL attempt yet. The
quality gates are in place with a documented baseline of exceptions for
pre-existing code (decision 0002); the burn-down tasks below remove them.

## Task list (dependency order; one session each)

Done criteria apply in addition to the standing rules: `make check` green,
tests for new behaviour, PR merged, this file updated.

### A. Structure (the "cleanup PR" series)

- [x] **A1 Move the pilots into the package.** Create
  `orbital_gym/autopilots/` (`base.py`, `orbit.py`, `moon.py`, `mars.py`,
  plus `lambert.py`), re-export from `scripts/autopilot_demo.py` so the
  CLI is unchanged. Done: `rl/moon_landing.py` imports from the package;
  the `ignore_imports` entry in `pyproject.toml` is deleted; import-linter
  passes without it. Depends on: nothing.
- [x] **A2 Explicit `Autopilot` hierarchy.** A base class owning
  `_act_world`/`_act_elev`, the gravity-turn ascent, hold and the
  suicide-burn `_brake`/`_descend` parameterised by a body; mission pilots
  override only their transfer logic. Done: `MoonAutopilot` and
  `MarsAutopilot` share the ascent code instead of copying it; all mission
  tests pass unchanged. Depends on A1.
- [x] **A3 `CelestialBody` abstraction.** A class (position, velocity,
  radius, mu, atmosphere, SOI, impact predictor, frame elements) used by
  `ToMoonEnv`, `ToMarsEnv`, the renderer and the pilots. Done: the
  `_moon_*` alias hooks in `ToMarsEnv` are gone; `rendering.py` draws a
  list of bodies; `ToMoon-v0`/`ToMars-v0` scores are unchanged (tests
  assert the pilots' terminal outcomes). Depends on A2.
- [x] **A4 `Guidance` class.** `conic_impact`, the zero-effort miss and
  velocity and the stop-and-drop virtual miss in one class built from a
  body and a state; used by the observation and the potential. Done:
  `MoonLandingEnv._guidance_obs` and `_potential` call it; the predictor
  test still matches brute-force coasting. Depends on A3.
- [x] **A5 Retire superseded trainers.** Delete `scripts/train_her.py`
  and `scripts/train_guided.py` (their diagnoses are in `EXPERIMENTS.md`);
  keep `train_ppo.py` as the plain baseline. Done: README no longer
  mentions them; their `pyproject.toml` baseline entries are deleted.
  Depends on: nothing.
- [x] **A6 Trainer defaults and `TrainingRun`.** The trainer lives in
  `orbital_gym/rl/trainer.py` (strict-typed): `TrainConfig` whose defaults
  are the v15 recipe, `BoundedPPO`, `Curriculum`, `PeriodicEval`,
  `TrainingRun`; `scripts/train_moon_rl.py` is an argparse front generated
  from the config's fields (`--r-<field>` for the reward). `--video` is on
  `fly_moon_policy.py` (`.mp4` through moviepy, `.gif` through imageio);
  `RecordVideo` does not fit because the scripted prefix runs in the raw
  env and the policy phase bypasses `reset()`. Done: a bare
  `python scripts/train_moon_rl.py --name x --hypothesis y` is the v15
  setup (`tests/test_trainer.py`); the file left the complexity and typing
  baselines; `--eval 8` on `best_100pct.zip` still lands 8/8. Depends on A5.
- [ ] **A7 HUD frame fix.** The `vr`/`vt` rows show body-relative values
  when inside a secondary body's SOI. Done: a render test asserts the HUD
  reads the moon-relative numbers after a moon landing. Depends on A3.

### B. The practice ladder

- [ ] **B1 Register `MoonLandingFromOrbit-v0`.** `MoonLandingEnv` as a
  Gymnasium id with a difficulty knob (`level`, `site_noise`,
  `extra_obs`), the state bank built on first use, and `info["success"]`
  on every env. Done: `gym.make` works from a fresh checkout; README
  documents it; smoke test covers it. Depends on A1.
- [ ] **B2 Register `MarsLandingFromOrbit-v0`.** Same wrapper over
  `ToMarsEnv` with a Mars bank from `MarsAutopilot`. Done: as B1. Depends
  on A3, B1.
- [ ] **B3 Benchmark protocol and baselines.** Fixed site grid and seeds
  per env; `scripts/benchmark.py` producing the table (random, autopilot,
  PPO); table in README with GIFs. Done: table reproducible by one
  command. Depends on B1.
- [ ] **B4 Camera zoom.** Keyboard, mouse wheel and on-screen buttons in
  `rendering.py`; exposed in `fly_manual.py`. Done: zoom persists across
  the rocket and map views; a render test exercises it. Depends on A3.

### C. v1 landing rules

- [ ] **C1 Landing legs and the touchdown hold.** `ToMoon-v1` and
  `MoonLandingFromOrbit-v1`: touchdown requires vertical speed, horizontal
  speed and tilt under limits, then 5 s on the surface with the engine
  effectively off (clock resets on lift-off); contact flag and hold
  progress in the observation; bonus paid after the hold. v0 untouched.
  Done: tests for each limit and for the reset; the autopilot still
  passes. Depends on A3, B1.
- [ ] **C2 Fine-tune the lander under v1.** Resume the v15 checkpoint on
  v1, log the run, report the 48-site result. Done: entry in
  `EXPERIMENTS.md` with the number. Depends on C1, A6.

### D. Publishing

- [ ] **D1 Package for PyPI.** Namespaced ids (`orbital_gym/…`), version,
  classifiers, `LICENSE`, a release workflow, the moon checkpoint as a
  release asset. Done: `pip install orbital-gym` in a clean venv passes
  the smoke test. Depends on B1.
- [ ] **D2 Gymnasium third-party listing and Hub baselines.** PR to the
  Gymnasium docs third-party page; SB3 baselines on the Hugging Face Hub;
  a Colab for the easy rungs. Done: links in README. Depends on B3, D1.

### E. Mars RL

- [ ] **E1 Mars landing from orbit.** Train the moon recipe on
  `MarsLandingFromOrbit-v0`. Done: logged run with a 48-site result.
  Depends on B2.
- [ ] **E2 Departure window and cruise.** Extend the curriculum to the
  parking orbit before departure; the Lambert residual as the guidance
  gauge. Done: logged run. Depends on E1.

### F. Baseline burn-down (one session each; delete the matching entry)

Each task: make the module pass `mypy --strict` (and the complexity limit
where listed) and delete its entry from `pyproject.toml`.

- [ ] F1 `orbital_gym/missions/to_orbit.py` (also `step`, `_substep`
  complexity)
- [ ] F2 `orbital_gym/rendering.py` (also `_orbit_path`, `_draw_hud`
  complexity)
- [ ] F3 `orbital_gym/missions/to_moon.py`, `to_mars.py`,
  `orbit_and_return.py` and `orbital_gym/rl/moon_landing.py`
- [ ] F4 `orbital_gym/autopilots/*` (also the `__call__` methods and
  `lambert` complexity; folds into A2)
- [ ] F5 `scripts/fly_manual.py`, `fly_moon_policy.py`, `train_ppo.py`
- [x] F6 `scripts/train_moon_rl.py` (done in A6)
- [ ] F7 `tests/test_env.py`, `test_orbit_and_return.py`, `test_to_moon.py`,
  `test_to_mars.py`

## Open Questions

Decided 2026-11-12 (owner): self-merge on green CI; repository made
public and protected (decision 0004 applied); typing ratchet as in
decision 0002; `uv.lock` kept; no publishing outside this repository
without the owner; landing-legs defaults 5 m/s / 3 m/s / 15 deg / 5 s
hold; background training runs allowed; tasks strictly in listed order.

1. The `edit` auto-commits on `master` came from a local tool; pushes of
   such commits are now rejected by branch protection. Point that tool at
   a branch or disable it for this repo.

## Next steps

1. A7 (HUD frame fix), then B1 onward in order.
