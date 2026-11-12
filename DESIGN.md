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
orbital_gym/guidance.py   Guidance: zero-effort landing instruments for a
                          body and a rocket state (conic impact, miss, stop
                          ratio), used by the RL observation and potential
orbital_gym/bodies.py     CelestialBody / CircularRails: moons, planets, stars
                          on rails (position, gravity, atmosphere, SOI,
                          elements, impact predictor)
orbital_gym/kinematics.py pure two-body functions
orbital_gym/presets.py    plain dataclasses: planets, moons, stars, rockets
```

Rules the import checker enforces:

- `orbital_gym.presets` and `orbital_gym.kinematics` import nothing else
  from the package; `orbital_gym.bodies` and `orbital_gym.guidance` import
  only those (guidance also bodies).
- `orbital_gym.missions` imports `presets`, `kinematics`, `bodies` (and
  `rendering` lazily, inside `render()`, which is the Gymnasium convention).
- `orbital_gym.rendering` imports only `presets`; it reads the env's
  `bodies` list through the `CelestialBody` interface.
- `orbital_gym.autopilots` imports `missions` and `presets`.
- `orbital_gym.rl` imports `missions`, `autopilots`, `presets`.
- Nothing inside `orbital_gym` imports `scripts` or `tests`.

All of these are enforced by import-linter contracts in `pyproject.toml`.

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

- **`CelestialBody`** (done, A3): position, velocity, gravity with the
  tidal correction, atmosphere, sphere of influence, elements and impact
  predictor for moons, planets and stars on `CircularRails`. Missions hold
  a `bodies` list and a `target_body`; the renderer and the pilots read
  bodies through that interface.
- **`Pilot` base class** (done, A2): the per-step `LocalState`, the
  act-in-world helpers, the shared ascent and hold; mission pilots override
  `_departure_ready` and `_transfer`. The landing logic reads its target
  through `self.body`.
- **`Guidance`** (done, A4): `conic_impact`, the zero-effort miss (actual
  and stop-and-drop virtual, blended through the burn), stop ratio and
  braking delta-v in one class built from a body and a state; the RL task's
  observation channels and shaping potential both read it.
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
