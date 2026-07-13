"""Scripted gravity-turn autopilot: proof that the environment is solvable.

Flies a classic ascent: vertical rise, pitch-over schedule down to the
horizon, cut the engine once apoapsis is high enough, coast, then burn
prograde near apoapsis until the periapsis is out of the atmosphere.

Run:  python scripts/autopilot_demo.py [--planet easy|earth] [--render]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import orbital_gym
from orbital_gym.two_stage_env import TwoStageToOrbitEnv, _wrap_pi


class Autopilot:
    """ascend -> coast -> insert -> done.

    Ascend flies an altitude-keyed pitch schedule until the apoapsis is above
    the atmosphere, coast rides up to apoapsis, and insert holds vertical
    speed near zero while burning horizontally until the periapsis clears the
    atmosphere. The vr-nulling insertion is what makes this work on any
    planet: on a big planet the apoapsis is reached with almost no horizontal
    velocity, so "circularization" is really the whole orbital insertion and
    cannot fit in a brief window near apoapsis.
    """

    def __init__(self, env: TwoStageToOrbitEnv):
        self.env = env
        self.mode = "ascend"

    def __call__(self, info: dict) -> np.ndarray:
        env, p = self.env, self.env.planet
        alt, apo, peri, vr = info["altitude"], info["apoapsis"], info["periapsis"], info["vr"]
        cutoff = p.space_altitude
        apo_target = 1.18 * cutoff
        peri_target = 1.03 * cutoff
        # Faster orbits (small planets) tolerate an early, aggressive
        # pitch-over; high-v_circ planets must stay lofted much longer.
        turn_end = cutoff * (0.75 + 0.35 * max(0.0, p.circular_velocity(cutoff) / 2300.0 - 1.0))
        prograde = math.atan2(vr, max(1.0, info["vt"]))  # flight path angle

        if self.mode == "ascend" and math.isfinite(apo) and apo >= apo_target:
            self.mode = "coast"
        if self.mode == "coast":
            if apo < 1.02 * cutoff:  # drag ate the apoapsis, keep climbing
                self.mode = "ascend"
            elif alt > max(1.02 * cutoff, 0.85 * apo) or (vr <= 0.0 and alt > 1.02 * cutoff):
                self.mode = "insert"
        if self.mode == "insert" and (
            peri >= peri_target or (peri > cutoff and peri >= alt - 500.0)
        ):
            self.mode = "done"

        if self.mode == "ascend":
            if alt < 250.0:
                pitch, throttle = math.pi / 2, 1.0
            else:
                f = min(1.0, (alt - 250.0) / (turn_end - 250.0))
                pitch = math.radians(88.0) * (1.0 - f) ** 0.65
                if vr < 40.0 and alt > 0.15 * cutoff:
                    # sinking while the apoapsis is still too low: pitch up,
                    # don't waste the burn flying level while falling
                    pitch = max(pitch, math.radians(30.0))
                throttle = 1.0
        elif self.mode == "insert":
            # Burn mostly horizontal while steering vertical speed toward an
            # altitude floor safely above the periapsis target -- prograde
            # burns can never lift the periapsis above the current radius.
            floor = 1.10 * cutoff
            vr_des = min(120.0, max(-40.0, 0.05 * (floor - alt)))
            pitch_max = 60.0 if alt < 1.02 * cutoff else 40.0
            pitch = math.radians(min(pitch_max, max(-20.0, 0.13 * (vr_des - vr))))
            throttle = 1.0
        else:  # coast / done
            pitch, throttle = prograde, 0.0

        err = _wrap_pi(pitch - math.radians(info["pitch_deg"]))
        turn = np.clip(err / (env.rocket.max_turn_rate * env.step_duration), -1.0, 1.0)
        return np.array([2.0 * throttle - 1.0, turn], dtype=np.float32)


def fly(env: TwoStageToOrbitEnv, verbose: bool = True):
    obs, info = env.reset()
    pilot = Autopilot(env)
    total_reward, last_print = 0.0, -1e9
    while True:
        action = pilot(info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if verbose and info["time"] - last_print >= 60.0:
            last_print = info["time"]
            apo = info["apoapsis"]
            apo_s = "   inf" if not math.isfinite(apo) else f"{apo / 1000:6.1f}"
            print(
                f"t={info['time']:6.0f}s  alt={info['altitude'] / 1000:6.1f}km "
                f"vt={info['vt']:5.0f}m/s  apo={apo_s}km peri={info['periapsis'] / 1000:7.1f}km "
                f"stage={info['stage'] + 1} dv_used={info['dv_used']:4.0f}m/s"
            )
        if terminated or truncated:
            return total_reward, info


def main():
    ap = argparse.ArgumentParser()
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

    env = TwoStageToOrbitEnv(**kwargs)
    print(orbital_gym.describe(env.rocket, env.planet))
    print("-" * 72)
    total_reward, info = fly(env)
    print("-" * 72)
    apo = info["apoapsis"]
    apo_s = "inf" if not math.isfinite(apo) else f"{apo / 1000:.1f} km"
    print(f"end: {info['termination']}  after {info['time']:.0f} s")
    print(f"orbit: {info['periapsis'] / 1000:.1f} x {apo_s} (cutoff {env.planet.space_altitude / 1000:.0f} km)")
    print(f"delta-v used {info['dv_used']:.0f} m/s of {env.total_delta_v():.0f} m/s budget "
          f"({info['dv_remaining']:.0f} m/s still in the tank)")
    print(f"episode return: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
