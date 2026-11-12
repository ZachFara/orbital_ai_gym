# orbital-gym design

## What this is

`orbital-gym` is a family of 2D rocket-mission environments for Gymnasium,
plus the reference pilots and the reinforcement-learning tooling that prove
they can be flown. The missions form a difficulty ladder on one physics
core: launch to orbit (`TwoStageToOrbit-v0`), orbit and return to the pad
(`OrbitAndReturn-v0`), a lunar landing with an on-rails moon (`ToMoon-v0`),
and a three-stage interplanetary flight to a Mars-like planet around a
star (`ToMars-v0`). Two things are deliverable to the outside world: the
environments as a pip package people can practice RL on, and the evidence
(scripted pilots that close every mission, learned policies with logged
experiments) that the ladder is solvable. The physics is deliberately
simple so the learning problem stays about orbital mechanics.

## Layers and module boundaries

Dependencies point downward only. Each layer may import the layers below
it and nothing above it.

```
scripts/            command-line entry points (train, fly, evaluate, log)
orbital_gym/rl/     RL task definitions on top of missions: start-state
                    banks, curricula, action-frame adapters, shaping
orbital_gym/autopilots/   scripted reference pilots (one class per mission)
orbital_gym/rendering.py  the pygame view; reads env state, imports nothing
                          from missions (duck-typed on the env)
orbital_gym/missions/     the environments: physics, reward, termination
orbital_gym/presets.py    plain dataclasses: planets, moons, stars, rockets
```

Rules the import checker enforces:

- `orbital_gym.presets` imports nothing else from the package.
- `orbital_gym.missions` imports only `presets` (and `rendering` lazily,
  inside `render()`, which is the Gymnasium convention).
- `orbital_gym.rendering` imports only `presets`.
- `orbital_gym.autopilots` imports `missions` and `presets`.
- `orbital_gym.rl` imports `missions`, `autopilots`, `presets`.
- Nothing inside `orbital_gym` imports `scripts` or `tests`.

The current tree has one known violation (the RL state bank imports the
pilots from `scripts/autopilot_demo.py`); moving the pilots into
`orbital_gym/autopilots/` is the first Phase 2 task.

## Core objects

- **Presets** (`Planet`, `Moon`, `Star`, `OrbitingBody`, `StarSystem`,
  `Stage`, `Rocket`): frozen dataclasses, SI units, no behaviour beyond
  small derived quantities. Every environment knob is a constructor
  argument, never a global.
- **`TwoStageToOrbitEnv`**: the physics core. Point-mass rocket, inverse-
  square gravity, exponential atmosphere, semi-implicit Euler substeps,
  staging, attitude as a rate-limited set-point. Missions extend it through
  three hooks: `_extra_acceleration` (more bodies), `_step_reward`
  (mission reward and mission terminals), `_obs` / `_info` (mission
  instruments). The base class owns `step()`, the turn penalty, the crash
  penalty, the escape guard (`_is_escaping`, overridable) and truncation.
- **Mission envs** (`OrbitAndReturnEnv`, `ToMoonEnv`, `ToMarsEnv`): phases
  with irreversible switches, potential-based progress rewards that cannot
  be farmed, and a landing-instrument panel in the observation. Spheres of
  influence switch the published elements to the body that matters.
- **Pilots**: one class per mission, a mode machine over the env's info,
  returning actions in the env's action space. They are proofs of
  solvability and generators of start states; they are never required at
  test time by a learned policy.
- **RL tasks** (`orbital_gym.rl`): a task wraps a mission env with a
  start-state distribution (a bank of expert states plus a curriculum
  level), an action-frame adapter, zero-effort guidance channels and a
  shaped training reward, while reporting the mission's native score.
- **Experiment log**: every training run is an MLflow run with
  hypothesis, outcome and diagnosis tags, mirrored into `EXPERIMENTS.md`.

## Object-oriented direction (Phase 2 targets)

The code grew mission by mission and has duplication that the next
missions would multiply. The intended shape:

- **`CelestialBody`** (position, velocity, radius, mu, optional
  atmosphere, sphere of influence) as the one abstraction for moons and
  planets on rails. `ToMoonEnv` and `ToMarsEnv` would hold a list of
  bodies; the renderer would draw bodies, not "the moon"; the impact
  predictor and frame switch would live on the body or a `Frame` helper
  instead of being copied between missions. This removes the
  `_moon_*` alias hooks `ToMarsEnv` currently exposes for the renderer.
- **`Autopilot` base class** owning the shared ascent, the act-in-world
  helpers and the suicide-burn descent, with mission pilots overriding
  only their transfer logic. `MarsAutopilot` already inherits from the
  moon pilot; the hierarchy should be explicit and the per-body landing
  logic parameterised by a `CelestialBody`.
- **Guidance instruments** (`conic_impact`, the zero-effort miss and
  velocity, the stop-and-drop virtual miss) as one `Guidance` class
  constructed from a body and a state, used by both the observation and
  the shaping potential instead of two code paths.
- **Trainer composition**: `BoundedPPO`, the curriculum callback, the
  probe evaluator and the experiment logger are already separate classes;
  the remaining `main()` logic should become a `TrainingRun` object so the
  CLI is a thin parser.
- Rules of thumb: a class per thing with state and behaviour; dataclasses
  for plain data; no abstraction before the second concrete use; no
  configuration nobody sets.

## Environment versioning

Published env ids are immutable benchmarks. A change to physics, reward or
termination that alters scores (landing legs and a touchdown hold, for
example) is a new id version (`-v1`); `-v0` keeps its numbers.

## Non-goals

Three-dimensional dynamics, torque and attitude dynamics, real planetary
ephemerides, and photorealism. Everything stays about guidance and orbital
mechanics in a plane.
