"""ToMoon: launch, circularize, hold orbit, then transfer to a lunar site.

This mission inherits the same ascent/orbit shaping as TwoStageToOrbit:
- potential-based climb reward
- in-space duration reward
- circularity-weighted orbit reward
- tiny turn penalty from the base step() path

After holding a stable orbit for `orbit_fraction` of a revolution (default half),
phase 2 begins and the reward switches to potential-based progress toward a
site on an orbiting moon -- and the base duration rewards stop, exactly as in
OrbitAndReturn: if space time kept paying, loitering to the time limit would
strictly beat landing (a trained policy demonstrated this before the fix).
A soft touchdown (relative speed <= moon_landing_speed) within
moon_site_radius of the site ends as "landed_moon_site" (+moon_site_bonus);
soft anywhere else on the moon ends as "landed_moon" (+moon_land_bonus);
anything harder is a crash. Third-body moon gravity is included via the base
mission's _extra_acceleration() hook.
"""

from __future__ import annotations

import math

import numpy as np

from ..presets import MOONS, Moon, Planet, Rocket
from .to_orbit import TWO_PI, TwoStageToOrbitEnv, _wrap_pi


class ToMoonEnv(TwoStageToOrbitEnv):
    OBS_SIZE = 15

    def __init__(
        self,
        planet: str | Planet = "easy",
        moon: str | Moon | None = None,
        rocket: str | Rocket | None = None,
        auto_stage: bool = True,
        step_duration: float = 1.0,
        physics_dt: float = 0.1,
        max_episode_steps: int = 7000,
        climb_reward: float = 100.0,
        space_rate: float = 1.0,
        orbit_rate: float = 1.0,
        orbit_ecc_scale: float = 0.25,
        turn_penalty: float = 0.01,
        crash_penalty: float = 100.0,
        crash_speed: float = 5.0,
        orbit_fraction: float = 0.5,
        hold_reward: float = 100.0,
        transfer_reward: float = 400.0,
        moon_site_bonus: float = 500.0,
        moon_land_bonus: float = 250.0,
        moon_site_radius: float = 8_000.0,
        moon_landing_speed: float = 25.0,
        render_mode: str | None = None,
    ):
        planet_name = planet if isinstance(planet, str) else planet.name
        if moon is None:
            moon = MOONS.get(planet_name, MOONS["easy"])
        self.moon = MOONS[moon] if isinstance(moon, str) else moon

        self.orbit_fraction = orbit_fraction
        self.hold_reward = hold_reward
        self.transfer_reward = transfer_reward
        self.moon_site_bonus = moon_site_bonus
        self.moon_land_bonus = moon_land_bonus
        self.moon_site_radius = moon_site_radius
        self.moon_landing_speed = moon_landing_speed

        self._required_angle = max(1e-6, orbit_fraction * TWO_PI)
        self._moon_d_scale = self.moon.orbit_radius

        super().__init__(
            planet=planet,
            rocket=rocket,
            auto_stage=auto_stage,
            step_duration=step_duration,
            physics_dt=physics_dt,
            max_episode_steps=max_episode_steps,
            climb_reward=climb_reward,
            space_rate=space_rate,
            orbit_rate=orbit_rate,
            orbit_ecc_scale=orbit_ecc_scale,
            turn_penalty=turn_penalty,
            crash_penalty=crash_penalty,
            crash_speed=crash_speed,
            terminate_on_orbit=False,
            render_mode=render_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._orbit_angle = 0.0
        self._transfer = False
        self._last_ang = math.pi / 2.0
        self._last_site_d = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self._last_site_d = self._moon_site_distance()
        return obs, info

    # --- moon kinematics ------------------------------------------------------

    def _moon_angle(self) -> float:
        return TWO_PI * (self._time / self.moon.orbit_period)

    def _moon_world(self) -> tuple[float, float]:
        a = self._moon_angle()
        return (
            self.moon.orbit_radius * math.cos(a),
            self.moon.orbit_radius * math.sin(a),
        )

    def _moon_velocity(self) -> tuple[float, float]:
        a = self._moon_angle()
        w = TWO_PI / self.moon.orbit_period
        r = self.moon.orbit_radius
        return (-w * r * math.sin(a), w * r * math.cos(a))

    def _moon_site_world(self) -> tuple[float, float]:
        mx, my = self._moon_world()
        a = self._moon_angle() + self.moon.site_angle
        return (
            mx + self.moon.radius * math.cos(a),
            my + self.moon.radius * math.sin(a),
        )

    def _moon_distance(self) -> float:
        mx, my = self._moon_world()
        return math.hypot(self._rx - mx, self._ry - my)

    def _moon_site_distance(self) -> float:
        sx, sy = self._moon_site_world()
        return math.hypot(self._rx - sx, self._ry - sy)

    def _moon_bearing(self) -> float:
        mx, my = self._moon_world()
        return _wrap_pi(math.atan2(my - self._ry, mx - self._rx) - math.atan2(self._ry, self._rx))

    def _moon_site_bearing(self) -> float:
        sx, sy = self._moon_site_world()
        return _wrap_pi(math.atan2(sy - self._ry, sx - self._rx) - math.atan2(self._ry, self._rx))

    # --- Physics hook ---------------------------------------------------------

    def _extra_acceleration(self, rx: float, ry: float, vx: float, vy: float) -> tuple[float, float]:
        mx, my = self._moon_world()
        dx = mx - rx
        dy = my - ry
        r2 = dx * dx + dy * dy
        # Softening avoids pathological accelerations if the agent intersects
        # the moon center in one step before collision handling runs.
        r2 = max(r2, (0.25 * self.moon.radius) ** 2)
        inv_r3 = 1.0 / (r2 * math.sqrt(r2))
        return self.moon.mu * dx * inv_r3, self.moon.mu * dy * inv_r3

    # --- Reward ---------------------------------------------------------------

    def _step_reward(self, alt_before: float, alt_after: float, space_time: float) -> float:
        # Impact handling against the moon surface, in any phase. A soft
        # touchdown anywhere on the moon is a landing (a smaller bonus off
        # the site); the crash penalty is applied by step() from _terminal.
        moon_d = self._moon_distance()
        if moon_d <= self.moon.radius:
            mvx, mvy = self._moon_velocity()
            rel_speed = math.hypot(self._vx - mvx, self._vy - mvy)
            site_d = self._moon_site_distance()
            if rel_speed > self.moon_landing_speed:
                self._terminal = "crashed"
                return 0.0
            if site_d <= self.moon_site_radius:
                self._terminal = "landed_moon_site"
                return self.moon_site_bonus
            self._terminal = "landed_moon"
            return self.moon_land_bonus

        ang = math.atan2(self._ry, self._rx)
        d_ang = abs(_wrap_pi(ang - self._last_ang))
        self._last_ang = ang

        if not self._transfer:
            reward = super()._step_reward(alt_before, alt_after, space_time)
            cutoff = self.planet.space_altitude
            peri, _, energy, ecc = self._orbit_elements()
            if alt_after > cutoff and energy < 0.0 and peri > cutoff:
                circ = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
                held_before = min(self._orbit_angle, self._required_angle)
                self._orbit_angle += d_ang * circ
                held_after = min(self._orbit_angle, self._required_angle)
                reward += self.hold_reward * (held_after - held_before) / self._required_angle
                if self._orbit_angle >= self._required_angle:
                    self._transfer = True
                    self._last_site_d = self._moon_site_distance()
            return reward

        # Transfer phase: potential-based progress toward the site only.
        # The base duration rewards must stop here, or loitering in orbit
        # to the time limit strictly beats landing (RL found this exploit
        # immediately: return 8722 by never landing at all).
        site_d = self._moon_site_distance()
        reward = self.transfer_reward * (self._last_site_d - site_d) / self._moon_d_scale
        self._last_site_d = site_d
        return reward

    # --- Derived quantities ---------------------------------------------------

    def _obs(self) -> np.ndarray:
        moon_b = self._moon_bearing()
        site_b = self._moon_site_bearing()
        progress = 1.0 if self._transfer else min(1.0, self._orbit_angle / self._required_angle)
        extras = np.array(
            [
                math.sin(moon_b),
                math.cos(moon_b),
                self._moon_distance() / self.moon.orbit_radius,
                math.sin(site_b),
                math.cos(site_b),
                progress,
            ],
            dtype=np.float32,
        )
        return np.concatenate([super()._obs(), extras])

    def _info(self):
        info = super()._info()
        mx, my = self._moon_world()
        sx, sy = self._moon_site_world()
        info["moon_x"] = mx
        info["moon_y"] = my
        info["moon_radius"] = self.moon.radius
        info["moon_distance"] = self._moon_distance()
        info["moon_bearing"] = self._moon_bearing()
        info["moon_site_x"] = sx
        info["moon_site_y"] = sy
        info["moon_site_distance"] = self._moon_site_distance()
        info["moon_site_bearing"] = self._moon_site_bearing()
        info["orbit_progress"] = min(1.0, self._orbit_angle / self._required_angle)
        info["transfer"] = self._transfer
        return info
