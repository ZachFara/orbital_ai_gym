"""Scripted autopilots: proof that the missions are solvable.

Ascent flies a classic gravity turn: vertical rise, pitch-over schedule down
to the horizon, cut the engine once apoapsis is high enough, coast, then burn
prograde near apoapsis until the periapsis is out of the atmosphere. The
return mission continues: hold orbit for the required half revolution, burn
retrograde at a fixed lead angle before the pad, fall through the atmosphere
(drag does most of the braking), and finish with a vertical landing burn.

The moon mission launches WESTWARD (the moon orbits counterclockwise, so an
eastward orbit meets it head-on at ~2 km/s closing speed), holds a low
parking orbit, fires trans-lunar injection when the phasing solution enters
its window, coasts out, and brakes retrograde into a soft off-site landing.
Putting the touchdown ON the site is deliberately left to RL
(train_guided.py starts its policies from this pilot's flights).

Run:  python scripts/autopilot_demo.py [--mission orbit|return|moon] [--planet easy|earth] [--render]
"""

from __future__ import annotations

import argparse
import math

import orbital_gym
from orbital_gym.autopilots import (
    Autopilot,
    MarsAutopilot,
    MoonAutopilot,
    ReturnAutopilot,
    lambert,
)
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_mars import ToMarsEnv
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv

__all__ = [
    "Autopilot",
    "MarsAutopilot",
    "MoonAutopilot",
    "ReturnAutopilot",
    "fly",
    "lambert",
    "main",
]


def fly(env: TwoStageToOrbitEnv, verbose: bool = True, pilot=None):
    _obs, info = env.reset()
    pilot = pilot or Autopilot(env)
    total_reward, last_print = 0.0, -1e9
    while True:
        action = pilot(info)
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if verbose and info["time"] - last_print >= 60.0:
            last_print = info["time"]
            apo = info["apoapsis"]
            apo_s = "   inf" if not math.isfinite(apo) else f"{apo / 1000:6.1f}"
            print(
                f"t={info['time']:6.0f}s  alt={info['altitude'] / 1000:6.1f}km "
                f"vt={info['vt']:5.0f}m/s  apo={apo_s}km peri={info['periapsis'] / 1000:7.1f}km "
                f"stage={info['stage'] + 1} dv_used={info['dv_used']:4.0f}m/s "
                f"[{pilot.mode}]"
            )
        if terminated or truncated:
            return total_reward, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="orbit", choices=["orbit", "return", "moon", "mars"])
    ap.add_argument("--planet", default="easy", choices=["easy", "earth"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="episode step limit")
    args = ap.parse_args()

    kwargs = dict(planet=args.planet)
    if args.planet == "earth":
        kwargs["max_episode_steps"] = 8000
    if args.steps:
        kwargs["max_episode_steps"] = args.steps
    if args.render:
        kwargs["render_mode"] = "human"

    if args.mission == "return":
        env = OrbitAndReturnEnv(**kwargs)
        pilot = ReturnAutopilot(env)
    elif args.mission == "moon":
        env = ToMoonEnv(**kwargs)
        pilot = MoonAutopilot(env)
    elif args.mission == "mars":
        kwargs.pop("planet", None)
        env = ToMarsEnv(**kwargs)
        pilot = MarsAutopilot(env)
    else:
        env = TwoStageToOrbitEnv(**kwargs)
        pilot = Autopilot(env)
    print(orbital_gym.describe(env.rocket, env.planet))
    print("-" * 72)
    total_reward, info = fly(env, pilot=pilot)
    print("-" * 72)
    apo = info["apoapsis"]
    apo_s = "inf" if not math.isfinite(apo) else f"{apo / 1000:.1f} km"
    print(f"end: {info['termination']}  after {info['time']:.0f} s")
    print(
        f"orbit: {info['periapsis'] / 1000:.1f} x {apo_s} (cutoff {env.planet.space_altitude / 1000:.0f} km)"
    )
    if "pad_downrange" in info:
        print(
            f"touchdown {info['pad_downrange'] / 1000:.2f} km from the pad "
            f"at {info['speed']:.1f} m/s"
        )
    if "mars_distance" in info:
        print(
            f"mars: {info['mars_altitude'] / 1000:.1f} km above the surface, "
            f"{info['mars_rel_speed']:.0f} m/s relative, "
            f"{info['mars_site_distance'] / 1000:.0f} km from the site"
        )
    elif "moon_distance" in info:
        mvx, mvy = env._moon_velocity()
        rel = math.hypot(env._vx - mvx, env._vy - mvy)
        print(
            f"moon: {(info['moon_distance'] - env.moon.radius) / 1000:.1f} km "
            f"above the surface, {rel:.0f} m/s relative, "
            f"{info['moon_site_distance'] / 1000:.0f} km from the site"
        )
    print(
        f"delta-v used {info['dv_used']:.0f} m/s of {env.total_delta_v():.0f} m/s budget "
        f"({info['dv_remaining']:.0f} m/s still in the tank)"
    )
    print(f"episode return: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
