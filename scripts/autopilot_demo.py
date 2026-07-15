"""Scripted autopilots: proof that the missions are solvable.

Ascent flies a classic gravity turn: vertical rise, pitch-over schedule down
to the horizon, cut the engine once apoapsis is high enough, coast, then burn
prograde near apoapsis until the periapsis is out of the atmosphere. The
return mission continues: hold orbit for the required half revolution, burn
retrograde at a fixed lead angle before the pad, fall through the atmosphere
(drag does most of the braking), and finish with a vertical landing burn.

Run:  python scripts/autopilot_demo.py [--mission orbit|return] [--planet easy|earth] [--render]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import orbital_gym
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv, _wrap_pi
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv


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

        return self._act(pitch, throttle, info)

    def _act(self, pitch: float, throttle: float, info: dict) -> np.ndarray:
        env = self.env
        err = _wrap_pi(pitch - math.radians(info["pitch_deg"]))
        turn = np.clip(err / (env.rocket.max_turn_rate * env.step_duration), -1.0, 1.0)
        return np.array([2.0 * throttle - 1.0, turn], dtype=np.float32)


class ReturnAutopilot(Autopilot):
    """The same ascent, then: hold -> deorbit -> entry -> land.

    Hold coasts prograde until the orbit-hold requirement is met and the pad
    is `lead` radians ahead. Deorbit burns retrograde until the periapsis is
    driven far underground (a steep entry keeps the ballistic downrange short
    and repeatable). Entry points up and lets drag do the braking; land flies
    a vr-vs-altitude braking profile down to a ~2 m/s touchdown while leaning
    the thrust toward the pad to fly out the last few km of error. The lead
    only needs to be right to a couple of km (the 1 s deorbit trigger
    quantizes at ~2 km anyway); the terminal steering does the rest.
    """

    def __init__(self, env: OrbitAndReturnEnv, lead: float = 0.65):
        super().__init__(env)
        self.lead = lead

    def __call__(self, info: dict) -> np.ndarray:
        if self.mode in ("ascend", "coast", "insert"):
            action = super().__call__(info)
            if self.mode != "done":
                return action
            self.mode = "hold"

        env, p = self.env, self.env.planet
        alt, vr, vt = info["altitude"], info["vr"], info["vt"]
        prograde = math.atan2(vr, max(1.0, vt))

        if self.mode == "hold":
            bearing = info["pad_bearing"]  # decreases toward 0 flying east
            if info["returning"] and 0.0 < bearing < self.lead:
                self.mode = "deorbit"
            else:
                return self._act(prograde, 0.0, info)
        if self.mode == "deorbit":
            if info["periapsis"] < -0.35 * p.radius:
                self.mode = "entry"
            else:
                return self._act(_wrap_pi(prograde + math.pi), 1.0, info)
        if self.mode == "entry" and alt < 8_000.0:
            self.mode = "land"
        if self.mode == "entry":
            return self._act(math.pi / 2.0, 0.0, info)

        # land: track a braking profile that reaches -2 m/s at the surface,
        # steering horizontal velocity toward the pad (bearing > 0 = pad is
        # "east" of us in the direction of decreasing polar angle, i.e. +vt)
        vr_des = -max(2.0, math.sqrt(10.0 * max(0.0, alt - 10.0)))
        a_cmd = p.surface_gravity + 1.0 * (vr_des - vr)
        thrust = env.rocket.stages[env._stage].thrust
        throttle = min(1.0, max(0.0, a_cmd * info["mass"] / thrust))
        vt_des = max(-70.0, min(70.0, 0.05 * info["pad_bearing"] * p.radius))
        pitch = math.radians(90.0 + max(-30.0, min(30.0, 0.5 * (vt - vt_des))))
        return self._act(pitch, throttle, info)


def fly(env: TwoStageToOrbitEnv, verbose: bool = True, pilot=None):
    obs, info = env.reset()
    pilot = pilot or Autopilot(env)
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
                f"stage={info['stage'] + 1} dv_used={info['dv_used']:4.0f}m/s "
                f"[{pilot.mode}]"
            )
        if terminated or truncated:
            return total_reward, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="orbit", choices=["orbit", "return"])
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
    print(f"orbit: {info['periapsis'] / 1000:.1f} x {apo_s} (cutoff {env.planet.space_altitude / 1000:.0f} km)")
    if "pad_downrange" in info:
        print(f"touchdown {info['pad_downrange'] / 1000:.2f} km from the pad "
              f"at {info['speed']:.1f} m/s")
    print(f"delta-v used {info['dv_used']:.0f} m/s of {env.total_delta_v():.0f} m/s budget "
          f"({info['dv_remaining']:.0f} m/s still in the tank)")
    print(f"episode return: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
