"""ToMars: launch, hold a parking orbit, escape the planet, cross the star
system on a transfer orbit, and land on a site on a Mars-like planet.

The "insane" rung. Everything ToMoon asks for, plus:

  * a **star** at the center of the system. The launch planet circles it on
    rails; so does the target. The simulation runs in the launch planet's
    (non-rotating, translating) frame, which the base env already assumes,
    and the star and the target enter as *tidal* accelerations: the pull on
    the rocket minus the pull on the planet. That is exact for a planet in
    free fall on a Keplerian circle, and it means the parking orbit is
    really perturbed by the star (the planet's Hill radius is ~2.7 planet
    radii in the "easy" system), the departure really is a hyperbolic
    escape, and the cruise really is a heliocentric conic.
  * a **three-stage rocket** (`ROCKETS["mars"]`, ~11.7 km/s).
  * a **departure window**: the target leads the planet by a phase angle that
    only works for a Hohmann-class transfer during a window; miss it and the
    next one is a planetary year away, far beyond the episode.
  * a target with a **thin atmosphere** (drag on the target-relative
    velocity), so a steep arrival is braked a little for free and a shallow
    one is not.

Phases and reward mirror ToMoon: the base climb / space / orbit shaping,
`hold_reward` for `orbit_fraction` of a revolution, then (duration rewards
off) `transfer_reward` paid as potential-based progress toward the landing
site, scaled by the target's orbit radius; `mars_site_bonus` for a soft
touchdown within `mars_site_radius`, `mars_land_bonus` anywhere else on the
target, the crash penalty otherwise. Flying into the star is a crash.
Spheres of influence are KSP-style Laplace radii: inside the target's the
published elements go target-relative; outside the planet's they go
heliocentric; the escape guard fires only on a star-hyperbolic trajectory
with an empty tank.

The renderer's secondary-body hooks were written for the moon; this env
exposes the target through the same names (`moon`, `_moon_world`, ...) so
the existing map view, SOI re-centering and HUD work unchanged, and adds
`_sun_world` / `_sun_velocity` / `planet_soi` for the system-wide view.
"""

from __future__ import annotations

import math

import numpy as np

from ..presets import PLANETS, ROCKETS, SYSTEMS, Planet, Rocket, StarSystem
from .to_orbit import TWO_PI, TwoStageToOrbitEnv, _elements, _wrap_pi


class ToMarsEnv(TwoStageToOrbitEnv):
    OBS_SIZE = 43

    def __init__(
        self,
        planet: str | Planet = "easy",
        system: str | StarSystem = "easy",
        rocket: str | Rocket | None = "mars",
        auto_stage: bool = True,
        step_duration: float = 1.0,
        physics_dt: float = 0.1,
        max_episode_steps: int = 18_000,
        climb_reward: float = 100.0,
        space_rate: float = 1.0,
        orbit_rate: float = 1.0,
        orbit_ecc_scale: float = 0.25,
        turn_penalty: float = 0.01,
        crash_penalty: float = 100.0,
        crash_speed: float = 5.0,
        orbit_fraction: float = 0.25,
        hold_reward: float = 100.0,
        transfer_reward: float = 400.0,
        mars_site_bonus: float = 500.0,
        mars_land_bonus: float = 250.0,
        mars_site_radius: float = 20_000.0,
        mars_landing_speed: float = 25.0,
        target_phase0: float | None = None,
        attitude_control: bool = True,
        hover_control: bool = False,
        render_mode: str | None = None,
    ):
        planet_obj = PLANETS[planet] if isinstance(planet, str) else planet
        self.system = SYSTEMS[system] if isinstance(system, str) else system
        self.star = self.system.star
        self.target = self.system.target
        # renderer compatibility: the target plays the role of the moon
        self.moon = self.target
        self.home_r = self.system.home_orbit_radius
        self.home_w = TWO_PI / self.system.home_orbit_period
        self.target_w = TWO_PI / self.target.orbit_period
        self.target_phase0 = (self.system.target_phase0 if target_phase0 is None
                              else float(target_phase0))
        self.planet_soi = self.home_r * (planet_obj.mu / self.star.mu) ** 0.4
        self.mars_soi = self.target.orbit_radius * (self.target.mu / self.star.mu) ** 0.4
        self.moon_soi = self.mars_soi
        self.sun_radius = self.star.radius
        self.v_home = math.sqrt(self.star.mu / self.home_r)

        self.orbit_fraction = orbit_fraction
        self.hold_reward = hold_reward
        self.transfer_reward = transfer_reward
        self.mars_site_bonus = mars_site_bonus
        self.mars_land_bonus = mars_land_bonus
        self.mars_site_radius = mars_site_radius
        self.mars_landing_speed = mars_landing_speed
        self.moon_landing_speed = mars_landing_speed
        self.moon_site_radius = mars_site_radius
        self._required_angle = max(1e-6, orbit_fraction * TWO_PI)
        self._d_scale = self.target.orbit_radius

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
            attitude_control=attitude_control,
            hover_control=hover_control,
            render_mode=render_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options and "site_angle" in options:
            self._site_angle = float(options["site_angle"])
        elif not hasattr(self, "_site_angle"):
            self._site_angle = self.target.site_angle
        if options and "target_phase0" in options:
            self.target_phase0 = float(options["target_phase0"])
        self._orbit_angle = 0.0
        self._transfer = False
        self._last_ang = math.pi / 2.0
        self._last_site_d = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self._last_site_d = self._mars_site_distance()
        return obs, info

    def set_site_angle(self, site_angle: float) -> None:
        self._site_angle = float(site_angle)

    # --- rails (all positions in the launch planet's frame) ------------------

    def _home_angle(self, t: float | None = None) -> float:
        t = self._time if t is None else t
        return self.system.home_phase0 + self.home_w * t

    def _mars_angle(self, t: float | None = None) -> float:
        t = self._time if t is None else t
        return self.target_phase0 + self.target_w * t

    _moon_angle = _mars_angle  # renderer / shared-helper name

    def _home_helio(self, t=None):
        a = self._home_angle(t)
        return self.home_r * math.cos(a), self.home_r * math.sin(a)

    def _home_helio_velocity(self, t=None):
        a = self._home_angle(t)
        s = self.home_w * self.home_r
        return -s * math.sin(a), s * math.cos(a)

    def _sun_world(self, t=None):
        hx, hy = self._home_helio(t)
        return -hx, -hy

    def _sun_velocity(self, t=None):
        vx, vy = self._home_helio_velocity(t)
        return -vx, -vy

    def _mars_helio(self, t=None):
        a = self._mars_angle(t)
        return (self.target.orbit_radius * math.cos(a),
                self.target.orbit_radius * math.sin(a))

    def _mars_world(self, t=None):
        mx, my = self._mars_helio(t)
        hx, hy = self._home_helio(t)
        return mx - hx, my - hy

    def _mars_velocity(self, t=None):
        a = self._mars_angle(t)
        s = self.target_w * self.target.orbit_radius
        hvx, hvy = self._home_helio_velocity(t)
        return -s * math.sin(a) - hvx, s * math.cos(a) - hvy

    def _mars_site_world(self, t=None):
        mx, my = self._mars_world(t)
        a = self._mars_angle(t) + self._site_angle
        return (mx + self.target.radius * math.cos(a),
                my + self.target.radius * math.sin(a))

    _moon_world = _mars_world
    _moon_velocity = _mars_velocity
    _moon_site_world = _mars_site_world

    def rocket_helio(self):
        """Rocket position and velocity relative to the star."""
        hx, hy = self._home_helio()
        hvx, hvy = self._home_helio_velocity()
        return self._rx + hx, self._ry + hy, self._vx + hvx, self._vy + hvy

    # --- geometry --------------------------------------------------------------

    def _mars_distance(self) -> float:
        mx, my = self._mars_world()
        return math.hypot(self._rx - mx, self._ry - my)

    def _mars_site_distance(self) -> float:
        sx, sy = self._mars_site_world()
        return math.hypot(self._rx - sx, self._ry - sy)

    def _sun_distance(self) -> float:
        sx, sy = self._sun_world()
        return math.hypot(self._rx - sx, self._ry - sy)

    def _in_mars_soi(self) -> bool:
        return self._mars_distance() <= self.mars_soi

    def _in_planet_soi(self) -> bool:
        return math.hypot(self._rx, self._ry) <= self.planet_soi

    _moon_distance = _mars_distance
    _moon_site_distance = _mars_site_distance
    _in_moon_soi = _in_mars_soi

    def soi(self) -> str:
        if self._in_mars_soi():
            return "mars"
        if self._in_planet_soi():
            return "planet"
        return "sun"

    def _mars_orbit_elements(self):
        mx, my = self._mars_world()
        mvx, mvy = self._mars_velocity()
        return _elements(self.target.mu, self.target.radius,
                         self._rx - mx, self._ry - my,
                         self._vx - mvx, self._vy - mvy)

    _moon_orbit_elements = _mars_orbit_elements

    def _helio_elements(self):
        x, y, vx, vy = self.rocket_helio()
        return _elements(self.star.mu, self.star.radius, x, y, vx, vy)

    def _mars_impact(self) -> tuple[bool, float]:
        """Where the current target-relative conic meets the target's surface:
        (impacting, signed angle from the site to that point). Same
        instrument as ToMoon's impact predictor."""
        mx, my = self._mars_world()
        mvx, mvy = self._mars_velocity()
        px, py = self._rx - mx, self._ry - my
        wx, wy = self._vx - mvx, self._vy - mvy
        mu, R = self.target.mu, self.target.radius
        r = math.hypot(px, py)
        theta_r = math.atan2(py, px)
        site = self._mars_angle() + self._site_angle
        if r <= R:
            return True, _wrap_pi(theta_r - site)
        h = px * wy - py * wx
        v2 = wx * wx + wy * wy
        rv = px * wx + py * wy
        ex = ((v2 - mu / r) * px - rv * wx) / mu
        ey = ((v2 - mu / r) * py - rv * wy) / mu
        ecc = math.hypot(ex, ey)
        if ecc < 1e-9 or abs(h) < 1.0:
            return abs(h) < 1.0, _wrap_pi(theta_r - site)
        p_param = h * h / mu
        f0 = math.copysign(
            math.acos(max(-1.0, min(1.0, (p_param / r - 1.0) / ecc))), rv)
        sgn = math.copysign(1.0, h)
        cos_fi = (p_param / R - 1.0) / ecc
        if cos_fi > 1.0:
            if 0.5 * v2 - mu / r >= 0.0 and rv > 0.0:
                return False, _wrap_pi(theta_r - site)
            return False, _wrap_pi(theta_r + sgn * (-f0 % TWO_PI) - site)
        fi = math.acos(max(-1.0, min(1.0, cos_fi)))
        return True, _wrap_pi(theta_r + sgn * ((-fi - f0) % TWO_PI) - site)

    _moon_impact = _mars_impact

    def _is_escaping(self) -> bool:
        # Planet-hyperbolic is the whole point of the transfer; only a
        # star-hyperbolic trajectory (and not inside the target's SOI, where
        # a landing may be imminent) counts as leaving for good.
        if self._in_mars_soi():
            return False
        return self._helio_elements()[2] >= 0.0

    # --- physics hook: star + target as tidal terms, target atmosphere -------

    def _extra_acceleration(self, rx: float, ry: float, vx: float, vy: float):
        ax = ay = 0.0
        for (bx, by), mu, soft in (
            (self._sun_world(), self.star.mu, self.star.radius),
            (self._mars_world(), self.target.mu, 0.25 * self.target.radius),
        ):
            dx, dy = bx - rx, by - ry
            r2 = max(dx * dx + dy * dy, soft * soft)
            inv = mu / (r2 * math.sqrt(r2))
            # minus the same body's pull on the planet at the origin
            b2 = bx * bx + by * by
            invb = mu / (b2 * math.sqrt(b2))
            ax += dx * inv - bx * invb
            ay += dy * inv - by * invb
        # target atmosphere: drag on the target-relative velocity
        if self.target.rho0 > 0.0:
            mx, my = self._mars_world()
            alt = math.hypot(rx - mx, ry - my) - self.target.radius
            if alt < self.target.space_altitude:
                rho = self.target.density(alt)
                mvx, mvy = self._mars_velocity()
                wx, wy = vx - mvx, vy - mvy
                speed = math.hypot(wx, wy)
                k = -0.5 * rho * self.rocket.stages[self._stage].cd_area * speed / self._mass
                ax += k * wx
                ay += k * wy
        return ax, ay

    # --- reward ----------------------------------------------------------------

    def _step_reward(self, alt_before: float, alt_after: float, space_time: float) -> float:
        if self._sun_distance() <= self.star.radius:
            self._terminal = "crashed"
            return 0.0
        mars_d = self._mars_distance()
        if mars_d <= self.target.radius:
            mvx, mvy = self._mars_velocity()
            rel_speed = math.hypot(self._vx - mvx, self._vy - mvy)
            if rel_speed > self.mars_landing_speed:
                self._terminal = "crashed"
                return 0.0
            if self._mars_site_distance() <= self.mars_site_radius:
                self._terminal = "landed_mars_site"
                return self.mars_site_bonus
            self._terminal = "landed_mars"
            return self.mars_land_bonus

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
                    self._last_site_d = self._mars_site_distance()
            return reward

        site_d = self._mars_site_distance()
        reward = self.transfer_reward * (self._last_site_d - site_d) / self._d_scale
        self._last_site_d = site_d
        return reward

    # --- observation -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        x, y, vx, vy = self.rocket_helio()
        r2 = self.target.orbit_radius
        hp, ha, henergy, hecc = self._helio_elements()
        vdir = math.atan2(vy, vx) if math.hypot(vx, vy) > 1.0 else self._home_angle() + math.pi / 2
        mhx, mhy = self._mars_helio()
        mars_b = _wrap_pi(math.atan2(mhy - y, mhx - x) - vdir)
        sx, sy = self._mars_site_world()
        hx, hy = self._home_helio()
        site_b = _wrap_pi(math.atan2(sy + hy - y, sx + hx - x) - vdir)
        progress = 1.0 if self._transfer else min(1.0, self._orbit_angle / self._required_angle)
        lead = _wrap_pi(self._mars_angle() - self._home_angle())

        # target-local instruments (the same panel ToMoon gives for the moon)
        mx, my = self._mars_world()
        mvx, mvy = self._mars_velocity()
        px, py = self._rx - mx, self._ry - my
        d = max(1.0, math.hypot(px, py))
        ux, uy = px / d, py / d
        vr_m = (self._vx - mvx) * ux + (self._vy - mvy) * uy
        vt_m = (self._vx - mvx) * uy - (self._vy - mvy) * ux
        malt = max(0.0, d - self.target.radius)
        pitch_m = _wrap_pi(self._theta - math.atan2(-ux, uy))
        dsite = _wrap_pi(self._mars_angle() + self._site_angle - math.atan2(py, px))
        g_m = self.target.mu / (d * d)
        a_max = self.rocket.stages[self._stage].thrust / self._mass
        need = (vr_m * vr_m + vt_m * vt_m) / (2.0 * max(malt, 1.0))
        tti = min(malt / max(1.0, -vr_m), 10_000.0)
        in_soi = d <= self.mars_soi
        impacting, miss = self._mars_impact() if in_soi else (False, 0.0)
        rho = self.target.density(malt)
        q = 0.5 * rho * (vr_m * vr_m + vt_m * vt_m)
        soi = self.soi()

        extras = np.array(
            [
                x / r2, y / r2,                               # heliocentric position
                vx / self.v_home, vy / self.v_home,           # heliocentric velocity
                min(3.0, max(0.0, (hp + self.star.radius) / r2)),   # helio periapsis / r2
                min(3.0, (ha + self.star.radius) / r2) if math.isfinite(ha) else 3.0,
                min(2.0, hecc),
                math.sin(mars_b), math.cos(mars_b),           # target bearing vs velocity
                min(3.0, self._mars_distance() / r2),
                math.sin(site_b), math.cos(site_b),
                progress,
                math.sin(lead), math.cos(lead),               # target's lead angle: the window
                math.log1p(malt / 1_000.0),
                min(malt, 20_000.0) / 20_000.0,
                vr_m / 1_000.0,
                vt_m / 1_000.0,
                math.sin(pitch_m), math.cos(pitch_m),
                math.sin(dsite), math.cos(dsite),
                self.delta_v_remaining() / self.total_delta_v(),
                min(2.0, g_m / a_max) / 2.0,
                min(2.0, need / max(a_max - g_m, 1e-3)) / 2.0,
                math.log1p(tti) / 9.0,
                1.0 if impacting else 0.0,
                math.sin(miss) if in_soi else 0.0,
                math.cos(miss) if in_soi else 0.0,
                1.0 if soi == "planet" else 0.0,
                1.0 if soi == "sun" else 0.0,
                1.0 if soi == "mars" else 0.0,
                min(1.0, q / 2_000.0),                        # dynamic pressure at the target
            ],
            dtype=np.float32,
        )
        return np.concatenate([super()._obs(), np.clip(extras, -20.0, 20.0)])

    def _info(self):
        info = super()._info()
        mx, my = self._mars_world()
        sx, sy = self._mars_site_world()
        mvx, mvy = self._mars_velocity()
        rel = math.hypot(self._vx - mvx, self._vy - mvy)
        d = self._mars_distance()
        info.update({
            "mars_x": mx, "mars_y": my, "mars_radius": self.target.radius,
            "mars_distance": d, "mars_altitude": d - self.target.radius,
            "mars_site_x": sx, "mars_site_y": sy,
            "mars_site_distance": self._mars_site_distance(),
            "mars_rel_speed": rel, "mars_soi": self.mars_soi,
            "planet_soi": self.planet_soi, "sun_distance": self._sun_distance(),
            "lead_angle": _wrap_pi(self._mars_angle() - self._home_angle()),
            "orbit_progress": min(1.0, self._orbit_angle / self._required_angle),
            "transfer": self._transfer,
            # renderer-compatible aliases
            "moon_x": mx, "moon_y": my, "moon_radius": self.target.radius,
            "moon_distance": d, "moon_rel_speed": rel, "moon_soi": self.mars_soi,
            "moon_site_distance": self._mars_site_distance(),
            "site_angle": self._site_angle,
        })
        impacting, miss = self._mars_impact()
        info["mars_impact"] = impacting
        info["mars_impact_miss"] = miss * self.target.radius
        soi = self.soi()
        info["soi"] = soi
        if soi == "mars":
            peri, apo, energy, ecc = self._mars_orbit_elements()
        elif soi == "sun":
            peri, apo, energy, ecc = self._helio_elements()
        else:
            peri = apo = energy = ecc = None
        if peri is not None:
            info["periapsis"] = peri
            info["apoapsis"] = apo
            info["eccentricity"] = ecc
            info["energy"] = energy
            info["circularity"] = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
            info["orbit"] = bool(energy < 0.0 and peri > 0.0)
        return info
