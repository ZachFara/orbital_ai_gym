# orbital-gym

A 2D **two-stage-to-orbit** [Gymnasium](https://gymnasium.farama.org/) environment.
Where LunarLander asks you to come down gently, this asks you to leave: fly a
two-stage rocket off the pad, out of the atmosphere, and into a stable orbit.

The reward is deliberately simple:

- **Below the atmosphere cutoff:** reward for altitude gained (sums to 100 for
  reaching the cutoff; potential-based, so hopping up and down can't farm it).
- **Above the cutoff:** +1 per second you stay up there.
- **In a stable orbit** (periapsis above the cutoff): up to +1 more per
  second, scaled by how circular the orbit is — a tidy circle pays double,
  a wild ellipse pays nothing extra.
- **Turning:** −0.01 per second at full turn command, a whisper of control
  cost so policies don't tumble end over end between burns.

The duration terms are the whole game. Fuel is finite, so hovering doesn't
pay and burning straight up ends in a fall. The only way to keep collecting
them is enough *horizontal* velocity to orbit — the reward function teaches
the gravity turn by itself. And without the circularity term, flinging
yourself onto a barely-bound ellipse with a huge apoapsis "stays up" exactly
as well as orbiting properly; with it, circularizing is strictly better.
Two guardrails keep it honest: reaching escape velocity with an empty tank
ends the episode (no reward for leaving forever), and crashing costs −100.

## Missions

One package, several registered missions sharing the physics, the rocket
presets, and the renderer:

| env ID | mission |
|---|---|
| `TwoStageToOrbit-v0` | the environment described above |
| `OrbitAndReturn-v0` | reach orbit, hold it half a revolution, then land back on the pad |
| `ToMoon-v0` | reach orbit, hold, then transfer to a site on an orbiting moon |

**ToMoon** adds a moon on rails (circular orbit, closed-form position) whose
gravity pulls on the rocket. After the same climb-and-hold as the return
mission, the reward switches to potential-based progress toward a landing
site on the moon's far face — and the duration rewards stop, exactly as in
the return mission (when they didn't, a trained policy proved that
loitering to the time limit strictly beat landing). Touching down at under
25 m/s relative speed pays +500 within 8 km of the site
(`landed_moon_site`), +250 anywhere else on the moon (`landed_moon`);
anything harder is a crash. One important quirk: the easy moon orbits
counterclockwise at 2.1 km/s — 73% faster than a Kepler orbit at its
radius — so missions must fly **westward** to chase it, and the delta-v
budget is razor thin (see below).

**OrbitAndReturn** is the boomerang mission, in two phases with an
irreversible switch:

1. **Up and hold:** the same climb reward, then `orbit_reward` (100) paid
   per radian of stable orbit swept, summing to exactly 100 once the hold is
   complete (`orbit_fraction=0.5` of a revolution). Progress accrues scaled
   by the same circularity measure, so a round orbit finishes the hold in
   exactly half a revolution and a sloppy ellipse takes proportionally
   longer. Deliberately angle-based, *not* per-second: a duration reward
   here would make "loiter in space until the time limit" strictly better
   than coming home.
2. **Come home:** the reward switches to potential-based progress toward the
   launch pad (100 for closing the full far-side distance; drifting away
   refunds it), plus a landing bonus of up to 300 that decays with touchdown
   distance (`exp(-d / 10 km)`) — and the usual −100 for crashing. Touch
   down softly within 500 m of the pad and the episode ends as
   `landed_on_pad`. A full success is worth roughly 600.

Its observation grows to 12 values: the base 9 plus sin/cos of the bearing
back to the pad and the hold progress (which doubles as the phase flag).

## Install

```bash
pip install -e ".[render,dev]"     # gymnasium + numpy, pygame, pytest
```

## Quickstart

```python
import gymnasium as gym
import orbital_gym

env = gym.make("TwoStageToOrbit-v0")   # or planet="earth" for hard mode
env = gym.make("OrbitAndReturn-v0")    # there and back again
obs, info = env.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["termination"], info["periapsis"], info["apoapsis"])
```

Watch the scripted autopilots fly the missions (proof the envs are solvable,
and a baseline for what "good" looks like):

```bash
python scripts/autopilot_demo.py --render           # gravity turn to orbit
python scripts/autopilot_demo.py --mission return   # ...and back: deorbit burn,
                                                    # reentry, landing burn onto the pad
python scripts/autopilot_demo.py --mission moon --render   # westward ascent, hold,
                                                    # trans-lunar injection, retro
                                                    # braking to a near-landing
python scripts/autopilot_demo.py --planet earth     # hard mode, ~11 km/s
```

The moon autopilot flies the whole profile — leaner westward ascent
(~4.07 km/s), Kepler-solved injection timing, midcourse coast, and a
retrograde suicide burn that kills the ~2.1 km/s arrival — but the rocket's
budget is ~100 m/s short of a soft touchdown: it runs the tank dry a few km
above the surface at a few hundred m/s. That last gap is deliberately left
to learning (below).

Fly it yourself (arrows steer, space toggles throttle, `s` stages, 1/2/3 time warp):

```bash
python scripts/fly_manual.py [--mission orbit|return|moon]
```

The flight view climbs with you: launch pad and hills at liftoff, layered sky
fading to stars, the jettisoned booster tumbling away at staging, and — once
your periapsis leaves the ground — a KSP-style pull-out to a whole-planet map
view with your predicted orbit ellipse and Ap/Pe markers.

Train PPO (needs `pip install -e ".[train]"`):

```bash
python scripts/train_ppo.py                    # 1M steps, saved under a fresh
                                               # timestamped name in models/
python scripts/train_ppo.py --mission return   # train the round trip
python scripts/train_ppo.py --watch models/<run>.zip
```

Or train **guided** by the scripted autopilots (much more sample-efficient,
and the only sane way to learn the moon mission):

```bash
python scripts/train_guided.py --mission moon          # or orbit / return
python scripts/train_guided.py --mission moon --eval 5 --watch models/<run>.zip
```

`train_guided.py` gives the policy three head starts: DART-style
demonstrations (autopilot rolled with small action noise, its clean
commands as labels), behavior cloning onto those demos, and a JSRL-style
jump-start curriculum — every episode begins with the autopilot flying a
random prefix of the mission before the policy takes over, and the maximum
prefix anneals to zero, so the policy masters the landing before the
launch. On the moon mission the policy starts from an expert that already
does everything but the final touchdown.

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
fraction, episode time fraction. `OrbitAndReturn-v0` appends sin/cos of the
pad bearing and the hold progress for `(12,)`.

`info` carries the physical truth every step: altitude, vr/vt, apoapsis,
periapsis, eccentricity, dynamic pressure, `dv_used`, `dv_remaining`,
`in_space`, `orbit`, and the termination reason — plus, on the return
mission, `pad_bearing`, `pad_distance`, `pad_downrange`, `orbit_progress`,
and `returning`.

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

The full round trip (`--mission return`, easy planet) costs the autopilot
~5.3 km/s: ascent, a ~660 m/s deorbit burn, a drag-braked reentry, and a
landing burn to a touchdown ~0.1 km from the pad.

The moon mission needs roughly all 6.9 km/s: ~4.07 ascent + ~0.57
injection + ~2.15 arrival braking leaves nothing for losses, which is why
the scripted pilot ends a few km short of touchdown (the easy moon's
on-rails speed is what makes it brutal: any arrival costs
`sqrt(v_inf² + v_esc²)` ≈ 2.15 km/s at the surface, v_inf ≥ 1.29 km/s
even on a perfect Hohmann). On `earth` the budget does not close at all.

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
    orbit_rate=1.0,            # bonus rate for a stable orbit, scaled by...
    orbit_ecc_scale=0.25,      # ...circularity: max(0, 1 - ecc / this)
    turn_penalty=0.01,         # tiny cost per second of turn command
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
