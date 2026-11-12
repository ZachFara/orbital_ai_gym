# PROGRESS

## Status (2026-11-12)

Phase 1 merged (PR #1); repository public with branch protection. Phase 2
in progress: A and B done. Four missions exist and are solved by scripted pilots;
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
- [x] **A7 HUD frame fix.** `Renderer._headline(body)` gives the HUD's
  altitude, speed, `vr` and `vt` in the frame of the body whose sphere of
  influence the rocket is in (`CelestialBody.radial_tangential`), the
  planet's otherwise. Done: `tests/test_rendering.py` flies a moon landing
  and asserts the moon-relative numbers against the env's planet-frame
  ones. Depends on A3.

### B. The practice ladder

- [x] **B1 Register `MoonLandingFromOrbit-v0`.** `MoonLandingFromOrbit`
  (`orbital_gym/rl/landing.py`) pins the reference reward, gamma,
  throttle map and guidance channels (decision 0008) and keeps `level`,
  `site_noise`, `extra_obs`, `nominal`; the bank builds on first use
  (~10 s). Every env reports `info["success"]` (`SUCCESS` per mission
  class). Done: `gym.make` works from a fresh checkout (CI builds the
  bank); README documents it; `tests/test_smoke.py` covers the id, its
  knobs and the flag. Depends on A1.
- [x] **B2 Register `MarsLandingFromOrbit-v0`.** The wrapper is now
  body-agnostic (`orbital_gym/rl/landing.py`: `LandingEnv` + a
  `LandingTask` per mission with env, orbit-first pilot, bank path,
  outcome names and body-scaled reward); `moon_landing.py` is a re-export
  shim so cached banks unpickle. `MarsAutopilot` already captures into a
  150 km parking orbit, so its from-orbit segment is the Mars bank
  (24 sites, ~60 s, horizon 20k steps because the deorbit wait can be a
  whole orbit). Done: `gym.make` works; README documents it;
  `tests/test_landing.py` + smoke cover it; the moon task is bit-identical
  (v15 `--eval 8` reproduces the same eight returns). Depends on A3, B1.
- [x] **B3 Benchmark protocol and baselines.** `orbital_gym/benchmark.py`
  (protocol, the three baselines, `run`, `table`) and `scripts/benchmark.py`
  (one command, `--ppo ENV=path`, `--record DIR` for GIFs); table and four
  GIFs in the README (decision 0009). Found and fixed on the way: the
  direct-descent moon pilot crashes at every site but the preset (the
  benchmark uses the orbit-first pilot), `TwoStageToOrbit-v0` reported no
  success when the pilot held orbit to the time limit (now success),
  `MarsLandingFromOrbit-v0` inherited the moon's 1,800-step window and
  timed out 11/16 (the window is now per task: 5,200). Done: table
  reproducible by one command; `tests/test_benchmark.py`. Depends on B1.
- [x] **B4 Camera zoom.** `Renderer.zoom` multiplies the automatic framing
  (`view_span`), so it persists across the rocket and map views;
  `handle_event` takes `+`/`-`/`0`, the mouse wheel and clicks on three
  on-screen buttons (drawn in human mode only, so recordings stay clean);
  `draw()` reads the mouse events itself, `fly_manual.py` forwards every
  event through the env's new `renderer` property. Done: a render test
  drives all three inputs and checks the zoom survives the map-view
  transition. Depends on A3.

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
  `orbit_and_return.py` and `orbital_gym/rl/landing.py`
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

1. C1 (landing legs and the 5 s hold as `-v1` envs), then C2 onward in order.
