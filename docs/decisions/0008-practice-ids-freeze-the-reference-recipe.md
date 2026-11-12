# 0008 Practice ids freeze the reference recipe

## Context

The learned moon landing was trained on `MoonLandingEnv`, a wrapper with
many knobs (reward weights, shaping gamma, throttle map, observation
channels, curriculum level, noise). The reference result (`ppo-zem-v15`,
`EXPERIMENTS.md`) is one point in that space. Publishing the task as a
Gymnasium id means picking one.

## Decision

- A practice id (`MoonLandingFromOrbit-v0`, `MarsLandingFromOrbit-v0`) is a
  thin subclass that pins the reference configuration: the reward
  (`REFERENCE_REWARD`, its distance/speed scales following the body), its
  shaping gamma, the exponential hover throttle and the guidance channels.
  Those are not constructor knobs of the id.
- The knobs it keeps are the practice ones: `level`, `site_noise`,
  `extra_obs`, `nominal`, `render_mode`, `seed`. Their defaults are the
  training distribution at full difficulty; `nominal=True` is the
  evaluation distribution.
- The trainer's `TrainConfig` reads the same constants, so "train with the
  defaults" and "train on the registered id" are the same task.
- Decision 0006 applies: once results are published against the id, any
  change to it ships as a new version.
- Every env (missions and practice ids) reports `info["success"]`: the
  mission's own notion of done right (`orbit`, `landed_on_pad`,
  `landed_moon_site`, `landed_mars_site`), so a benchmark script can treat
  them uniformly.

## Consequences

- Reward-shaping research keeps using `MoonLandingEnv` directly; the id
  is for practice and benchmarking.
- The expert state bank is built on first `gym.make` (about ten seconds)
  and cached under `cache/`, which is gitignored.
