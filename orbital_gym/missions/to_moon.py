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

Inside the moon's sphere of influence (Laplace radius) the frame of
reference switches, KSP-style: info's periapsis/apoapsis/eccentricity
become moon-relative, and the escape guard is suspended -- planet-centric
elements say nothing about where a rocket deep in the moon's gravity is
actually headed (on a fast-railed moon a lander braking onto the surface
is outright planet-hyperbolic, and the guard used to call it "escaped"
moments before touchdown the instant its tank ran dry).
"""

from __future__ import annotations

import math

import numpy as np

from ..presets import MOONS, PLANETS, Moon, Planet, Rocket
from .to_orbit import TWO_PI, TwoStageToOrbitEnv, _elements, _wrap_pi


class ToMoonEnv(TwoStageToOrbitEnv):
    OBS_SIZE = 30

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
        planet_obj = PLANETS[planet] if isinstance(planet, str) else planet
        if moon is None:
            moon = MOONS.get(planet_obj.name, MOONS["easy"])
        self.moon = MOONS[moon] if isinstance(moon, str) else moon
        if rocket is None and planet_obj.name == "easy":
            rocket = "moon"  # the ~10 km/s preset: the mission's real budget

        # KSP-style sphere of influence (Laplace radius). Inside it the
        # elements in info, the escape logic, and the renderer all switch
        # to the moon's frame -- a rocket braking onto the moon is planet-
        # hyperbolic, and judging it by planet-centric elements is nonsense.
        self.moon_soi = self.moon.orbit_radius * (self.moon.mu / planet_obj.mu) ** 0.4

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
        # The landing site is settable per-episode (goal-conditioned training
        # randomizes it): options={"site_angle": rad} overrides the preset,
        # otherwise the preset holds. Dynamics never depend on the site -- only
        # the reward, the obs site channels, and the impact predictor do -- so
        # randomizing it leaves the cached expert-action prefix exactly valid.
        if options and "site_angle" in options:
            self._site_angle = float(options["site_angle"])
        elif not hasattr(self, "_site_angle"):
            self._site_angle = self.moon.site_angle
        self._orbit_angle = 0.0
        self._transfer = False
        self._last_ang = math.pi / 2.0
        self._last_site_d = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self._last_site_d = self._moon_site_distance()
        return obs, info

    def set_site_angle(self, site_angle: float) -> None:
        """Move the landing site (moon-body-frame angle, rad). Persists until
        the next reset that overrides it. Dynamics are unaffected."""
        self._site_angle = float(site_angle)

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
        a = self._moon_angle() + self._site_angle
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

    def _in_moon_soi(self) -> bool:
        return self._moon_distance() <= self.moon_soi

    def _moon_orbit_elements(self):
        """(periapsis_alt, apoapsis_alt, energy, ecc) relative to the moon."""
        mx, my = self._moon_world()
        mvx, mvy = self._moon_velocity()
        return _elements(self.moon.mu, self.moon.radius,
                         self._rx - mx, self._ry - my,
                         self._vx - mvx, self._vy - mvy)

    def _moon_impact(self) -> tuple[bool, float]:
        """Where the current moon-relative conic meets the moon.

        Returns (impacting, miss): whether the coasting trajectory ahead of
        the rocket reaches the surface, and the signed angle around the moon
        from the landing site to that surface point (to the next periapsis
        point when it doesn't reach). Two-body prediction in the moon frame:
        exact for the terminal descent, bent by the planet's tide over long
        coasts -- an instrument, not an oracle.
        """
        mx, my = self._moon_world()
        mvx, mvy = self._moon_velocity()
        px, py = self._rx - mx, self._ry - my
        wx, wy = self._vx - mvx, self._vy - mvy
        mu, R = self.moon.mu, self.moon.radius
        r = math.hypot(px, py)
        theta_r = math.atan2(py, px)
        site = self._moon_angle() + self._site_angle
        if r <= R:
            return True, _wrap_pi(theta_r - site)
        h = px * wy - py * wx
        v2 = wx * wx + wy * wy
        rv = px * wx + py * wy
        ex = ((v2 - mu / r) * px - rv * wx) / mu
        ey = ((v2 - mu / r) * py - rv * wy) / mu
        ecc = math.hypot(ex, ey)
        if ecc < 1e-9 or abs(h) < 1.0:
            # circular never lands; a radial drop lands straight below
            return abs(h) < 1.0, _wrap_pi(theta_r - site)
        p_param = h * h / mu
        f0 = math.copysign(
            math.acos(max(-1.0, min(1.0, (p_param / r - 1.0) / ecc))), rv)
        sgn = math.copysign(1.0, h)
        cos_fi = (p_param / R - 1.0) / ecc
        if cos_fi > 1.0:
            # periapsis clears the surface: no impact ahead
            if 0.5 * v2 - mu / r >= 0.0 and rv > 0.0:
                return False, _wrap_pi(theta_r - site)  # outbound hyperbola
            return False, _wrap_pi(theta_r + sgn * (-f0 % TWO_PI) - site)
        # the conic dips below r=R at true anomaly -fi (rising back out at
        # +fi); the next crossing forward of the rocket's anomaly f0 is impact
        fi = math.acos(max(-1.0, min(1.0, cos_fi)))
        return True, _wrap_pi(theta_r + sgn * ((-fi - f0) % TWO_PI) - site)

    def _is_escaping(self) -> bool:
        # Never declare an escape inside the moon's SOI: the rocket may be
        # about to land (or gravity-assist back down). Once it leaves the
        # SOI, the planet-centric check applies again and still closes the
        # burn-to-infinity exploit.
        if self._in_moon_soi():
            return False
        return super()._is_escaping()

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

        # Moon-local instruments: the quantities a landing actually controls.
        # The base obs is planet-framed; without these the network has to
        # derive its state relative to a moon moving at 2.1 km/s (and the
        # final 20 km of altitude vanish into the fourth decimal place of
        # the orbit-radius-normalized distance).
        mx, my = self._moon_world()
        mvx, mvy = self._moon_velocity()
        px, py = self._rx - mx, self._ry - my
        d = max(1.0, math.hypot(px, py))
        ux, uy = px / d, py / d
        vr_m = (self._vx - mvx) * ux + (self._vy - mvy) * uy
        vt_m = (self._vx - mvx) * uy - (self._vy - mvy) * ux
        malt = max(0.0, d - self.moon.radius)
        pitch_m = _wrap_pi(self._theta - math.atan2(-ux, uy))
        dsite = _wrap_pi(self._moon_angle() + self._site_angle
                         - math.atan2(py, px))

        # Powered-landing instruments: the numbers a lander's guidance
        # computer would display, so the network reads them instead of
        # deriving them from raw state.
        g_m = self.moon.mu / (d * d)
        a_max = self.rocket.stages[self._stage].thrust / self._mass
        need = (vr_m * vr_m + vt_m * vt_m) / (2.0 * max(malt, 1.0))
        tti = min(malt / max(1.0, -vr_m), 10_000.0)
        in_soi = d <= self.moon_soi
        impacting, miss = self._moon_impact() if in_soi else (False, 0.0)

        extras = np.array(
            [
                math.sin(moon_b),
                math.cos(moon_b),
                d / self.moon.orbit_radius,
                math.sin(site_b),
                math.cos(site_b),
                progress,
                math.log1p(malt / 1_000.0),      # moon altitude, log: resolves
                min(malt, 20_000.0) / 20_000.0,  # ...600 km and 600 m alike;
                                                 # linear over the last 20 km
                vr_m / 1_000.0,                  # moon-relative sink/climb rate
                vt_m / 1_000.0,                  # moon-relative ground speed
                math.sin(pitch_m),               # attitude above the MOON's
                math.cos(pitch_m),               # local horizon
                math.sin(dsite),                 # site angle around the moon,
                math.cos(dsite),                 # from the rocket's position
                self.delta_v_remaining() / self.total_delta_v(),
                min(2.0, g_m / a_max) / 2.0,     # throttle that hovers
                min(2.0, need / max(a_max - g_m, 1e-3)) / 2.0,
                                                 # suicide-burn margin: decel
                                                 # to stop at the surface vs
                                                 # headroom; 0.5 = last moment
                math.log1p(tti) / 9.0,           # time to surface at this
                                                 # sink rate (log-scaled)
                1.0 if impacting else 0.0,       # coasting path hits the moon
                math.sin(miss) if in_soi else 0.0,
                math.cos(miss) if in_soi else 0.0,
                                                 # predicted touchdown vs the
                                                 # site, angle around the moon
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
        info["site_angle"] = self._site_angle
        info["moon_site_x"] = sx
        info["moon_site_y"] = sy
        info["moon_site_distance"] = self._moon_site_distance()
        info["moon_site_bearing"] = self._moon_site_bearing()
        info["orbit_progress"] = min(1.0, self._orbit_angle / self._required_angle)
        info["transfer"] = self._transfer
        mvx, mvy = self._moon_velocity()
        info["moon_rel_speed"] = math.hypot(self._vx - mvx, self._vy - mvy)
        info["moon_soi"] = self.moon_soi
        impacting, miss = self._moon_impact()
        info["moon_impact"] = impacting
        # signed surface arc from the site to the predicted touchdown point
        info["moon_impact_miss"] = miss * self.moon.radius
        if self._in_moon_soi():
            # KSP-style frame switch: inside the SOI the published elements
            # are moon-relative (the planet-centric ones are meaningless).
            peri, apo, energy, ecc = self._moon_orbit_elements()
            info["soi"] = "moon"
            info["periapsis"] = peri
            info["apoapsis"] = apo
            info["eccentricity"] = ecc
            info["energy"] = energy
            info["circularity"] = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
            info["orbit"] = bool(energy < 0.0 and peri > 0.0)
        else:
            info["soi"] = "planet"
        return info
