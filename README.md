# orbital-gym

A 2D **two-stage-to-orbit** [Gymnasium](https://gymnasium.farama.org/) environment.
Where LunarLander asks you to come down gently, this asks you to leave: fly a
two-stage rocket off the pad, out of the atmosphere, and into a stable orbit.

The reward is deliberately simple:

- **Below the atmosphere cutoff:** reward for altitude gained (sums to 100 for
  reaching the cutoff; potential-based, so hopping up and down can't farm it).
- **Above the cutoff:** +1 per second you stay up there.

That second term is the whole game. Fuel is finite, so hovering doesn't pay
and burning straight up ends in a fall. The only way to keep collecting the
duration reward is enough *horizontal* velocity to orbit — the reward
function teaches the gravity turn by itself. Two guardrails keep it honest:
reaching escape velocity with an empty tank ends the episode (no reward for
leaving forever), and crashing costs −100.

## Install

```bash
pip install -e ".[render,dev]"     # gymnasium + numpy, pygame, pytest
```

## Quickstart

```python
import gymnasium as gym
import orbital_gym

env = gym.make("TwoStageToOrbit-v0")   # or planet="earth" for hard mode
obs, info = env.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["termination"], info["periapsis"], info["apoapsis"])
```

Watch a scripted gravity turn fly to orbit (proof the env is solvable, and a
baseline for what "good" looks like):

```bash
python scripts/autopilot_demo.py --render        # small planet, ~4.3 km/s to orbit
python scripts/autopilot_demo.py --planet earth  # hard mode, ~11 km/s
```

Fly it yourself (arrows steer, space toggles throttle, `s` stages, 1/2/3 time warp):

```bash
python scripts/fly_manual.py
```

The flight view climbs with you: launch pad and hills at liftoff, layered sky
fading to stars, the jettisoned booster tumbling away at staging, and — once
your periapsis leaves the ground — a KSP-style pull-out to a whole-planet map
view with your predicted orbit ellipse and Ap/Pe markers.

Train PPO (needs `pip install -e ".[train]"`):

```bash
python scripts/train_ppo.py --steps 3_000_000
python scripts/train_ppo.py --watch models/ppo_orbit.zip
```

## Spaces

**Action** — `Box(-1, 1, (2,))`, or `(3,)` with `auto_stage=False`:

| index | meaning |
|---|---|
| 0 | throttle, mapped to 0..1 |
| 1 | turn rate, mapped to ±15°/s |
| 2 | (manual staging only) stage when > 0 |

**Observation** — `Box((9,), float32)`, normalized: altitude / cutoff, radial
velocity, tangential velocity (both / orbital velocity), sin & cos of pitch
above the local horizon, current-stage fuel fraction, stage index, mass
fraction, episode time fraction.

`info` carries the physical truth every step: altitude, vr/vt, apoapsis,
periapsis, eccentricity, dynamic pressure, `dv_used`, `dv_remaining`,
`in_space`, `orbit`, and the termination reason.

## Presets and difficulty

The default planet `"easy"` is Kerbin-scale: Earth surface gravity but a
600 km radius, so orbital velocity is ~2.3 km/s and one orbit takes ~31 min —
episodes stay short (default 4000 steps × 1 s). `"earth"` is the real thing.

Delta-v is known up front (Tsiolkovsky — `orbital_gym.describe(rocket, planet)`
prints the mission briefing):

| planet | needs (incl. losses) | rocket carries | autopilot actually used |
|---|---|---|---|
| easy  | ~3.3 km/s | 6.9 km/s | 4.3 km/s → 75×90 km orbit |
| earth | ~9.7 km/s | 13.1 km/s | 11.0 km/s → 103×3653 km orbit |

Make it easier or harder by constructing the env directly: planets and
rockets are plain dataclasses (`Planet`, `Rocket`, `Stage` in
`orbital_gym.presets`), and every knob is a constructor argument:

```python
from orbital_gym import TwoStageToOrbitEnv

env = TwoStageToOrbitEnv(
    planet="easy",             # name, or a Planet(...) of your own
    rocket=None,               # defaults to the preset matched to the planet
    auto_stage=True,           # False = agent must fire staging itself
    terminate_on_orbit=False,  # True = end early with the coasting-equivalent
                               # bonus once periapsis clears the atmosphere
                               # (reward-identical, much faster training)
    max_episode_steps=4000,
    climb_reward=100.0, space_rate=1.0, crash_penalty=100.0,
)
```

## Physics (and honest simplifications)

Point-mass rocket, inverse-square gravity, exponential atmosphere with
quadratic drag, semi-implicit Euler at 0.1 s substeps (energy drift over a
full orbit < 0.5%, covered by a test). Thrust is throttleable 0–100% with
unlimited relights; attitude is rate-limited but massless (no torque
dynamics); no planet rotation, no max-Q structural limit, no Isp altitude
dependence. All deliberate, to keep the learning problem about *orbital
mechanics*.

Ideas for later: planet rotation (free eastward boost), discrete
LunarLander-style actions, max-Q constraint, wind, a third stage, fuel-optimal
reward shaping.
