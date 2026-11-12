"""Scripted pilots for TwoStageToOrbit and OrbitAndReturn: gravity-turn
ascent, coast, insertion, and for the return mission the hold, deorbit,
reentry and landing burn back onto the pad."""

from __future__ import annotations

import math

import numpy as np

from ..missions.orbit_and_return import OrbitAndReturnEnv
from ..missions.to_orbit import TwoStageToOrbitEnv, _wrap_pi
from .base import throttle_action


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
        p = self.env.planet
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
        if env.attitude_control:
            # pitch is elevation above the local horizon, so the desired
            # direction is sin(pitch) radial + cos(pitch) tangential
            return np.array(
                [throttle_action(self.env, throttle), math.sin(pitch), math.cos(pitch)],
                dtype=np.float32,
            )
        err = _wrap_pi(pitch - math.radians(info["pitch_deg"]))
        turn = np.clip(err / (env.rocket.max_turn_rate * env.step_duration), -1.0, 1.0)
        return np.array([throttle_action(self.env, throttle), turn], dtype=np.float32)


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

    def __init__(self, env: OrbitAndReturnEnv, lead: float = 0.79):
        # lead tuned on the "easy" planet by scanning: the touchdown point
        # moves ~575 km east per radian of extra lead, on-pad near 0.78-0.79
        # with the terminal steering absorbing the last couple of km.
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
                # hold fire until the 180-degree slew to retrograde is nearly
                # done; burning mid-slew pushes prograde and raises the orbit
                retro = _wrap_pi(prograde + math.pi)
                aligned = abs(_wrap_pi(retro - math.radians(info["pitch_deg"]))) < 0.26
                return self._act(retro, 1.0 if aligned else 0.0, info)
        if self.mode == "entry" and alt < 12_000.0:
            self.mode = "land"
        if self.mode == "entry":
            return self._act(math.pi / 2.0, 0.0, info)

        # land: track a braking profile that reaches -2 m/s at the surface,
        # steering horizontal velocity toward the pad (bearing > 0 = pad is
        # "east" of us in the direction of decreasing polar angle, i.e. +vt).
        # While the horizontal error is large, slow the descent and translate:
        # hovering costs ~10 m/s of delta-v per second, and there is over a
        # km/s to spare, so trading fuel for landing accuracy is the deal.
        x_err = info["pad_bearing"] * p.radius
        if info["dv_remaining"] < 350.0 or abs(x_err) > 20_000.0:
            x_err = 0.0  # not enough fuel (or too far) to chase the pad: land soft here
        vt_des = max(-90.0, min(90.0, 0.08 * x_err))
        slow = 1.0 / (1.0 + abs(x_err) / 2_500.0)
        vr_des = -max(2.0, slow * math.sqrt(10.0 * max(0.0, alt - 10.0)))
        a_cmd = p.surface_gravity + 1.0 * (vr_des - vr)
        thrust = env.rocket.stages[env._stage].thrust
        throttle = min(1.0, max(0.0, a_cmd * info["mass"] / thrust))
        pitch = math.radians(90.0 + max(-35.0, min(35.0, 0.5 * (vt - vt_des))))
        return self._act(pitch, throttle, info)
