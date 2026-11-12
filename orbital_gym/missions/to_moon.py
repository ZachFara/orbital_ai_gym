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
from typing import Any

import numpy as np
from gymnasium import spaces

from ..bodies import CelestialBody
from ..presets import MOONS, PLANETS, Moon, Planet, Rocket
from ..touchdown import TouchdownRules
from .to_orbit import TWO_PI, TwoStageToOrbitEnv, _wrap_pi


class ToMoonEnv(TwoStageToOrbitEnv):
    SUCCESS = ("landed_moon_site",)
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
        attitude_control: bool = True,
        hover_control: bool = False,
        render_mode: str | None = None,
        touchdown: TouchdownRules | None = None,
    ):
        planet_obj = PLANETS[planet] if isinstance(planet, str) else planet
        if moon is None:
            moon = MOONS.get(planet_obj.name, MOONS["easy"])
        self.moon = MOONS[moon] if isinstance(moon, str) else moon
        if rocket is None and planet_obj.name == "easy":
            rocket = "moon"  # the ~10 km/s preset: the mission's real budget

        # The moon as a body on rails about the planet at the origin. Its
        # KSP-style sphere of influence (Laplace radius) is where the elements
        # in info, the escape logic and the renderer switch to the moon's
        # frame -- a rocket braking onto the moon is planet-hyperbolic, and
        # judging it by planet-centric elements is nonsense.
        self.moon_body = CelestialBody.from_preset("moon", "moon", self.moon, planet_obj.mu)
        self.target_body = self.moon_body
        self.bodies = [self.moon_body]
        self.moon_soi = self.moon_body.soi

        self.orbit_fraction = orbit_fraction
        self.hold_reward = hold_reward
        self.transfer_reward = transfer_reward
        self.moon_site_bonus = moon_site_bonus
        self.moon_land_bonus = moon_land_bonus
        self.moon_site_radius = moon_site_radius
        self.moon_landing_speed = moon_landing_speed

        self._required_angle = max(1e-6, orbit_fraction * TWO_PI)
        self._moon_d_scale = self.moon.orbit_radius

        # v1 landing legs (ToMoon-v1): None reproduces v0 exactly
        self.touchdown = touchdown
        self._contact = False
        self._hold = 0.0
        self._contact_phi = 0.0
        self._touchdown_limit: str | None = None
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
        if touchdown is not None:
            self.observation_space = spaces.Box(
                -20.0, 20.0, shape=(self.OBS_SIZE + 2,), dtype=np.float32
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
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
        self._contact = False
        self._hold = 0.0
        self._contact_phi = 0.0
        self._touchdown_limit = None
        obs, info = super().reset(seed=seed, options=options)
        self._last_site_d = self._moon_site_distance()
        return obs, info

    def set_site_angle(self, site_angle: float) -> None:
        """Move the landing site (moon-body-frame angle, rad). Persists until
        the next reset that overrides it. Dynamics are unaffected."""
        self._site_angle = float(site_angle)

    # --- moon kinematics: thin names over the body ----------------------------

    def _moon_angle(self) -> float:
        return self.moon_body.angle(self._time)

    def _moon_world(self) -> tuple[float, float]:
        return self.moon_body.position(self._time)

    def _moon_velocity(self) -> tuple[float, float]:
        return self.moon_body.velocity(self._time)

    def _moon_site_world(self) -> tuple[float, float]:
        return self.moon_body.site_world(self._time, self._site_angle)

    def _moon_distance(self) -> float:
        return self.moon_body.distance(self._rx, self._ry, self._time)

    def site_distance(self) -> float:
        """Straight-line distance (m) from the rocket to the landing site."""
        return self._moon_site_distance()

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
        return self.moon_body.in_soi(self._rx, self._ry, self._time)

    def _moon_orbit_elements(self):
        """(periapsis_alt, apoapsis_alt, energy, ecc) relative to the moon."""
        return self.moon_body.elements(self._rx, self._ry, self._vx, self._vy, self._time)

    def _moon_impact(self) -> tuple[bool, float]:
        """(impacting, signed angle from the site to the predicted touchdown):
        the body's two-body impact predictor, see `CelestialBody.impact`."""
        return self.moon_body.impact(
            self._rx, self._ry, self._vx, self._vy, self._time, self._site_angle
        )

    def _is_escaping(self) -> bool:
        # Never declare an escape inside the moon's SOI: the rocket may be
        # about to land (or gravity-assist back down). Once it leaves the
        # SOI, the planet-centric check applies again and still closes the
        # burn-to-infinity exploit.
        if self._in_moon_soi():
            return False
        return super()._is_escaping()

    # --- v1: landing legs and the touchdown hold --------------------------------

    def _substep(self, throttle: float, turn_rate: float) -> None:
        super()._substep(throttle, turn_rate)
        if self.touchdown is not None and self._terminal is None:
            self._legs_contact()

    def _legs_frame(self) -> tuple[float, float, float, float, float, float]:
        """(vr, vt, tilt, a_thrust, g_m, moon_d): the moon-relative radial and
        tangential speeds, the tilt off the local vertical, the thrust
        acceleration, local gravity and the distance to the moon's centre."""
        mx, my = self._moon_world()
        mvx, mvy = self._moon_velocity()
        px, py = self._rx - mx, self._ry - my
        d = max(1.0, math.hypot(px, py))
        ux, uy = px / d, py / d
        vr = (self._vx - mvx) * ux + (self._vy - mvy) * uy
        vt = (self._vx - mvx) * uy - (self._vy - mvy) * ux
        tilt = _wrap_pi(self._theta - math.atan2(uy, ux))
        a_thrust = self._throttle_actual * self.rocket.stages[self._stage].thrust / self._mass
        return vr, vt, tilt, a_thrust, self.moon.mu / (d * d), d

    def _legs_contact(self) -> None:
        """Per physics substep: contact against the limits, lift-off, and
        resting on the legs."""
        rules = self.touchdown
        assert rules is not None
        vr, vt, tilt, a_thrust, g_m, d = self._legs_frame()
        if not self._contact:
            if d > self.moon.radius:
                return
            self._touchdown_limit = rules.violated(vr, vt, tilt)
            if self._touchdown_limit is not None:
                self._terminal = "crashed"
                return
            self._contact = True
            self._hold = 0.0
            mx, my = self._moon_world()
            self._contact_phi = _wrap_pi(
                math.atan2(self._ry - my, self._rx - mx) - self._moon_angle()
            )
            self._settle()
            return
        if a_thrust * math.cos(tilt) > g_m:
            # lift-off: free flight resumes (the physics already moved it) and
            # the hold clock resets
            self._contact = False
            self._hold = 0.0
            return
        self._settle()

    def _legs_step(self, moon_d: float) -> float | None:
        """Per step, after the physics: the hold clock while on the legs and
        the landing once it runs out. None while airborne (the normal shaping
        applies); 0.0 for a crash the substeps found."""
        rules = self.touchdown
        assert rules is not None
        if self._terminal == "crashed":
            return 0.0
        if not self._contact:
            return None
        _vr, _vt, _tilt, a_thrust, g_m, _d = self._legs_frame()
        if a_thrust < rules.engine_off_fraction * g_m:
            self._hold += self.step_duration
        else:
            self._hold = 0.0  # the engine came on: start the hold over
        if self._hold >= rules.hold_seconds:
            if self._moon_site_distance() <= self.moon_site_radius:
                self._terminal = "landed_moon_site"
                return self.moon_site_bonus
            self._terminal = "landed_moon"
            return self.moon_land_bonus
        return 0.0

    def _settle(self) -> None:
        """Rest on the legs: the contact bearing on the surface, carried round
        with the (tidally locked) moon, at rest in the moon-centred frame.
        Every speed in this env (the v0 limit, the instruments, the HUD) is
        measured against the moon's centre, not the turning ground, so a
        lander at rest reads zero and lifts off with no sideways kick."""
        a = self._moon_angle() + self._contact_phi
        mx, my = self._moon_world()
        radius = self.moon.radius
        self._rx, self._ry = mx + radius * math.cos(a), my + radius * math.sin(a)
        self._vx, self._vy = self._moon_velocity()

    # --- Physics hook ---------------------------------------------------------

    def _extra_acceleration(
        self, rx: float, ry: float, vx: float, vy: float
    ) -> tuple[float, float]:
        return self.moon_body.gravity(rx, ry, self._time)

    # --- Reward ---------------------------------------------------------------

    def _step_reward(self, alt_before: float, alt_after: float, space_time: float) -> float:
        # Impact handling against the moon surface, in any phase. A soft
        # touchdown anywhere on the moon is a landing (a smaller bonus off
        # the site); the crash penalty is applied by step() from _terminal.
        moon_d = self._moon_distance()
        if self.touchdown is not None:
            legs = self._legs_step(moon_d)
            if legs is not None:
                return legs
        elif moon_d <= self.moon.radius:
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
        dsite = _wrap_pi(self._moon_angle() + self._site_angle - math.atan2(py, px))

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
                math.log1p(malt / 1_000.0),  # moon altitude, log: resolves
                min(malt, 20_000.0) / 20_000.0,  # ...600 km and 600 m alike;
                # linear over the last 20 km
                vr_m / 1_000.0,  # moon-relative sink/climb rate
                vt_m / 1_000.0,  # moon-relative ground speed
                math.sin(pitch_m),  # attitude above the MOON's
                math.cos(pitch_m),  # local horizon
                math.sin(dsite),  # site angle around the moon,
                math.cos(dsite),  # from the rocket's position
                self.delta_v_remaining() / self.total_delta_v(),
                min(2.0, g_m / a_max) / 2.0,  # throttle that hovers
                min(2.0, need / max(a_max - g_m, 1e-3)) / 2.0,
                # suicide-burn margin: decel
                # to stop at the surface vs
                # headroom; 0.5 = last moment
                math.log1p(tti) / 9.0,  # time to surface at this
                # sink rate (log-scaled)
                1.0 if impacting else 0.0,  # coasting path hits the moon
                math.sin(miss) if in_soi else 0.0,
                math.cos(miss) if in_soi else 0.0,
                # predicted touchdown vs the
                # site, angle around the moon
            ],
            dtype=np.float32,
        )
        if self.touchdown is None:
            return np.concatenate([super()._obs(), extras])
        legs = np.array(
            [
                1.0 if self._contact else 0.0,  # on the legs
                min(1.0, self._hold / self.touchdown.hold_seconds),  # hold progress
            ],
            dtype=np.float32,
        )
        return np.concatenate([super()._obs(), extras, legs])

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
        if self.touchdown is not None:
            info["contact"] = self._contact
            info["hold_progress"] = min(1.0, self._hold / self.touchdown.hold_seconds)
            info["touchdown_limit"] = self._touchdown_limit
        return info
