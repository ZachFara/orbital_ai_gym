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
| `ToMars-v0` | three stages: escape the planet, cross a star system, land on a Mars-like planet |

**ToMoon** adds a moon on rails (circular orbit, closed-form position) whose
gravity pulls on the rocket. After the same climb-and-hold as the return
mission, the reward switches to potential-based progress toward a landing
site on the moon's far face — and the duration rewards stop, exactly as in
the return mission (when they didn't, a trained policy proved that
loitering to the time limit strictly beat landing). Touching down at under
25 m/s relative speed pays +500 within 8 km of the site
(`landed_moon_site`), +250 anywhere else on the moon (`landed_moon`);
anything harder is a crash. The moon's on-rails period is **Keplerian**
for its orbit radius — this matters more than it sounds: rails at any
other speed make the moon-relative frame non-inertial, and a lander
fights a phantom `v²/r − μ/r²` force the whole way down (an earlier
arbitrary period put 1.2 m/s², 14% of moon surface gravity, under every
landing). The moon orbits counterclockwise, so missions fly **westward**
to arrive co-rotating instead of meeting it head-on at ~2 km/s.

The moon has a KSP-style **sphere of influence** (the Laplace radius,
~820 km for the easy pair). Inside it, `info`'s periapsis / apoapsis /
eccentricity switch to moon-relative values, and the escape guard is
suspended — planet-centric elements say nothing about where a rocket
deep in the moon's gravity is actually headed (on a fast-railed moon a
braking lander is outright planet-hyperbolic, and the guard used to call
it "escaped" seconds before touchdown the moment its tank ran dry). The
renderer follows suit: in the moon's SOI the map view re-centers on the
moon, the trajectory ellipse is drawn around it, and the HUD's altitude
and speed go moon-relative.

**ToMars** is the insane rung. The launch planet now circles a **star**
on rails, and so does a Mars-like target with a thin atmosphere; the
rocket is a three-stage ~11.7 km/s vehicle (`ROCKETS["mars"]`). The
simulation stays in the launch planet's frame, with the star and the
target entering as *tidal* accelerations (their pull on the rocket minus
their pull on the planet), which is exact for a planet in free fall on a
Keplerian circle. The system is compact so the trip fits an episode
(~15,000 one-second steps): the planet's year is 3.9 h, a Hohmann
transfer 2.8 h, and the planet is 0.35% of the star's mass. That compactness
is what makes it hard: the star's tide is 3.4% of gravity in the parking
orbit (the Hill radius is three planet radii), it does about a kilometre
per second of work on the rocket during the climb-out, so patched conics
cannot predict the heliocentric orbit after escape, and the departure
window closes in minutes and does not come back within the episode.
Reward mirrors ToMoon: the climb and hold, then potential-based progress
toward the landing site scaled by the target's orbit radius, +500 for a
soft touchdown within 20 km of the site, +250 anywhere else on the target,
−100 for crashing (into the star too). Spheres of influence switch the
published elements between planet, star and target; the escape guard
fires only on a star-hyperbolic coast with an empty tank. The observation
is `(43,)`: the base 9, heliocentric position/velocity/elements, target
and site bearings, the target's lead angle (the departure window), a
target-local landing panel identical to the moon's, the SOI as a one-hot,
and the dynamic pressure in the target's atmosphere.

The scripted `MarsAutopilot` proves it closes: ascent and a quarter-orbit
hold, a simulate-ahead departure (for a grid of burns it coasts the full
force model out of the planet's influence and prices the rest with a
free-arrival-time Lambert solve, firing when burn plus correction is
cheap), Lambert midcourse corrections aimed at a 150 km periapsis, capture
into a parking orbit, a timed deorbit and the moon pilot's powered
descent. It lands at ~1 m/s about 100 km from the site with ~3 km/s to
spare, in ~15,300 steps.

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
                                                    # braking to a soft landing
python scripts/autopilot_demo.py --planet earth     # hard mode, ~11 km/s
python scripts/autopilot_demo.py --mission mars --render   # three stages, a star,
                                                    # a departure window, a Mars
```

The moon autopilot flies the whole profile — leaner westward ascent
(~4.07 km/s), Kepler-solved injection timing, midcourse coast, a
retrograde suicide burn that kills the ~1.8 km/s arrival, and a
boundary-riding vertical descent to a touchdown at the profile's ~8 m/s
floor, with ~3.4 km/s to spare. It lands ~287 km from the site: putting
the landing *on* the site is deliberately left to learning (below).

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
python scripts/train_guided.py --mission moon --resume models/<run>.zip
python scripts/train_guided.py --mission moon --eval 5 --watch models/<run>.zip
```

`train_guided.py` gives the policy three head starts: DART-style
demonstrations (autopilot rolled with small action noise, its clean
commands as labels), behavior cloning onto those demos, and a JSRL-style
jump-start curriculum — every episode begins with the autopilot flying a
random prefix of the mission before the policy takes over, and the maximum
prefix anneals to zero, so the policy masters the landing before the
launch. On the moon mission the policy starts from an expert that already
soft-lands — a couple hundred km off target: reaching the site (and flying
it cheaper) is what's left to learn.

`--resume models/<run>.zip` keeps training an existing model for `--steps`
more steps, skipping the demo and cloning stages. Each run saves its
reward-normalization statistics next to the model
(`<run>_vecnorm.pkl`) so the resumed value function sees the same reward
scale it trained on; older checkpoints without the file resume fine, the
stats just re-warm.

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

`ToMoon-v0`'s observation is `(30,)`: the base 9, six moon/site bearings
and distances, and — because the landing is flown relative to a moving
moon — a panel of moon-local instruments: log-scaled and last-20-km moon
altitude, moon-relative sink rate and ground speed, attitude above the
moon's local horizon, the site's angle around the moon, the delta-v
fraction remaining, and the readouts a lander's guidance computer would
show — the throttle that hovers, the suicide-burn margin (deceleration
needed to stop at the surface vs. available headroom), log time-to-surface,
and an **impact predictor**: whether the coasting trajectory hits the
moon, plus sin/cos of the angle from the predicted touchdown point to the
site. The predictor converges on the actual landing point ~11 minutes
before touchdown, so aiming for the site becomes reading a gauge instead
of deriving orbital mechanics.

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

The moon mission flies a dedicated ~10 km/s rocket (`ROCKETS["moon"]`,
the `ToMoon-v0` default): arrival costs `sqrt(v_inf² + v_esc²)` ≈
1.8 km/s at the surface (the moon's own 1.73 km/s escape velocity
dominates; v_inf ≈ 0.4 km/s on the transfer), and on the 6.9 km/s shared
rocket the ~4.07 ascent + ~0.57 injection + arrival braking left almost
nothing for descent losses — the mission closed only on a perfect
flight, if at all. The moon rocket keeps the same liftoff TWR and
stage-1 burn time so the same ascent flies, and its big vacuum stage
pays for injection, braking, and the landing with ~3.4 km/s to spare.
On `earth` the budget still does not close at all.

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
