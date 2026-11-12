"""OrbitAndReturn: reach a stable orbit, hold it, then land back on the pad.

Same world, rocket, physics, and action space as TwoStageToOrbit. The mission
has two phases with different rewards, switching irreversibly the moment the
rocket has swept `orbit_fraction` of a revolution (default half) around the
planet while in a stable orbit (periapsis above the atmosphere):

Phase 1 -- get up and hold orbit:
    climb   the same potential-based climb reward as TwoStageToOrbit
            (sums to climb_reward for reaching the atmosphere cutoff).
    orbit   orbit_reward, paid per radian of stable orbit swept, summing to
            exactly orbit_reward when the required angle is complete.
            Deliberately angle-based, not time-based: a per-second reward
            would make "loiter in space until the time limit" strictly
            better than coming home, and would pay more for slower, more
            eccentric orbits. Progress accrues scaled by the same
            circularity measure the base mission rewards,
            max(0, 1 - ecc / orbit_ecc_scale): a circular orbit completes
            the hold in exactly the required angle, a sloppy ellipse takes
            proportionally longer, and ecc >= orbit_ecc_scale doesn't
            count as holding orbit at all.

Phase 2 -- come home:
    descent potential-based on the straight-line distance to the launch pad
            at (0, R): descent_reward * d(distance) / (2 R). Getting closer
            pays, drifting away refunds, so it cannot be farmed.
    landing on a soft touchdown (impact below crash_speed), a bonus of
            land_bonus * exp(-downrange / land_decay), where downrange is the
            surface arc distance to the pad. Touching down within pad_radius
            terminates as "landed_on_pad" instead of "landed".
    crash   -crash_penalty, as in the base mission.

No reward accrues for staying in space after the phase switch, so with any
discounting the only thing left worth doing is flying home.

Observation: the base 9 values plus
    9   sin(pad bearing)   bearing = polar angle of the rocket minus the
    10  cos(pad bearing)   pad's angle; 0 means directly above the pad
    11  orbit-hold progress, min(1, swept / required); 1 means phase 2

Termination adds "landed_on_pad" to the base reasons. terminate_on_orbit is
not available here (orbit is the midpoint of this mission, not the end).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..presets import Planet, Rocket
from .to_orbit import TWO_PI, TwoStageToOrbitEnv, _wrap_pi


class OrbitAndReturnEnv(TwoStageToOrbitEnv):
    SUCCESS = ("landed_on_pad",)
    OBS_SIZE = 12

    def __init__(
        self,
        planet: str | Planet = "easy",
        rocket: str | Rocket | None = None,
        auto_stage: bool = True,
        step_duration: float = 1.0,
        physics_dt: float = 0.1,
        max_episode_steps: int = 5000,
        climb_reward: float = 100.0,
        crash_penalty: float = 100.0,
        crash_speed: float = 5.0,
        orbit_fraction: float = 0.5,
        orbit_reward: float = 100.0,
        orbit_ecc_scale: float = 0.25,
        turn_penalty: float = 0.01,
        descent_reward: float = 100.0,
        land_bonus: float = 300.0,
        land_decay: float = 10_000.0,
        pad_radius: float = 500.0,
        attitude_control: bool = True,
        hover_control: bool = False,
        render_mode: str | None = None,
    ):
        # Mission state used by reset()/_obs(), which run inside super().__init__.
        self.orbit_fraction = orbit_fraction
        self.orbit_reward = orbit_reward
        self.descent_reward = descent_reward
        self.land_bonus = land_bonus
        self.land_decay = land_decay
        self.pad_radius = pad_radius
        self._required_angle = orbit_fraction * TWO_PI
        super().__init__(
            planet=planet,
            rocket=rocket,
            auto_stage=auto_stage,
            step_duration=step_duration,
            physics_dt=physics_dt,
            max_episode_steps=max_episode_steps,
            climb_reward=climb_reward,
            orbit_ecc_scale=orbit_ecc_scale,
            turn_penalty=turn_penalty,
            crash_penalty=crash_penalty,
            crash_speed=crash_speed,
            terminate_on_orbit=False,
            attitude_control=attitude_control,
            hover_control=hover_control,
            render_mode=render_mode,
        )
        self._d_scale = 2.0 * self.planet.radius

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._orbit_angle = 0.0
        self._returning = False
        self._last_ang = math.pi / 2.0  # polar angle of the pad
        self._last_pad_d = 0.0
        return super().reset(seed=seed, options=options)

    # --- Reward ---------------------------------------------------------------

    def _step_reward(self, alt_before: float, alt_after: float, space_time: float) -> float:
        ang = math.atan2(self._ry, self._rx)
        d_ang = abs(_wrap_pi(ang - self._last_ang))
        self._last_ang = ang

        if not self._returning:
            reward = self._climb_term(alt_before, alt_after)
            cutoff = self.planet.space_altitude
            peri, _, energy, ecc = self._orbit_elements()
            if alt_after > cutoff and energy < 0.0 and peri > cutoff:
                circ = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
                held_before = min(self._orbit_angle, self._required_angle)
                self._orbit_angle += d_ang * circ
                held_after = min(self._orbit_angle, self._required_angle)
                reward += self.orbit_reward * (held_after - held_before) / self._required_angle
                if self._orbit_angle >= self._required_angle:
                    self._returning = True
                    self._last_pad_d = self._pad_distance()
            return reward

        d = self._pad_distance()
        reward = self.descent_reward * (self._last_pad_d - d) / self._d_scale
        self._last_pad_d = d
        if self._terminal == "landed":
            downrange = self._pad_downrange()
            reward += self.land_bonus * math.exp(-downrange / self.land_decay)
            if downrange <= self.pad_radius:
                self._terminal = "landed_on_pad"
        return reward

    # --- Derived quantities -----------------------------------------------------

    def _pad_bearing(self) -> float:
        return _wrap_pi(math.atan2(self._ry, self._rx) - math.pi / 2.0)

    def _pad_distance(self) -> float:
        """Straight-line distance to the pad at (0, R)."""
        return math.hypot(self._rx, self._ry - self.planet.radius)

    def _pad_downrange(self) -> float:
        """Arc distance along the surface to the pad."""
        return self.planet.radius * abs(self._pad_bearing())

    def _obs(self) -> np.ndarray:
        bearing = self._pad_bearing()
        progress = min(1.0, self._orbit_angle / self._required_angle)
        extra = np.array([math.sin(bearing), math.cos(bearing), progress], dtype=np.float32)
        return np.concatenate([super()._obs(), extra])

    def _info(self):
        info = super()._info()
        info["pad_bearing"] = self._pad_bearing()
        info["pad_distance"] = self._pad_distance()
        info["pad_downrange"] = self._pad_downrange()
        info["orbit_progress"] = min(1.0, self._orbit_angle / self._required_angle)
        info["returning"] = self._returning
        return info
