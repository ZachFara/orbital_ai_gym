"""Scripted pilot for ToMars: the moon pilot's ascent, hold, capture and
descent, plus simulate-ahead departure planning through the star's tide,
free-arrival-time Lambert midcourse corrections and an offset aim point
for a parking-orbit capture at the target."""

from __future__ import annotations

import math

import numpy as np

from ..missions.to_mars import ToMarsEnv
from ..missions.to_orbit import TWO_PI, _wrap_pi
from .lambert import lambert
from .moon import MoonAutopilot


class MarsAutopilot(MoonAutopilot):
    """ascend -> coast -> inject -> hold -> depart -> cruise -> brake -> descend.

    The ascent, parking orbit, braking and descent are the moon pilot's,
    with the target standing in for the moon. What is new is the middle:

    * **depart**: in the parking orbit, every step solves the departure for
      a grid of hyperbolic excess speeds. For each: the burn point whose
      outgoing asymptote is parallel to the planet's heliocentric velocity
      (prograde escape), the time to leave the planet's SOI, the heliocentric
      transfer from the exit point to the target's orbit radius, and the
      phase gap between where the target will be and where the transfer
      arrives. Fire when some candidate has both a small gap and the rocket
      at its burn point; burn prograde until that excess speed is reached.
    * **cruise**: coast on the heliocentric conic. While far from the target,
      trim the arrival phase gap with small prograde/retrograde burns (the
      derivative of the gap with respect to speed is taken numerically).
      Near the target, hand over to the zero-effort-miss divert, then to the
      moon pilot's retro brake and suicide-profile descent.

    Tuned for the "easy" star system; lands soft a few hundred km from the
    site, like the moon pilot. Putting it on the site is left to learning.
    """

    GAP_TOL = math.radians(1.0)  # arrival phase gap accepted at departure
    PERI_FRAC = 1.5  # higher parking orbit: the star's tide wobbles
    LEVEL_FRAC = 1.52  # the periapsis by ~20 km, keep it clear of the air
    MCC_BUDGET = 1_800.0  # m/s of midcourse correction allowed: the star's
    # tide does ~1 km/s of work during the climb-out,
    # so the heliocentric orbit after escape is not
    # the patched-conic one and must be re-targeted
    MCC_MIN = 1.0  # m/s: smaller Lambert residuals are left alone
    DEPART_MAX = 1_700.0  # m/s: depart when burn + predicted correction is under this
    REPLAN_GAIN = 100.0  # adopt a new arrival plan only if it saves this much
    REPLAN_RESIDUAL = 150.0  # ...or the residual to the current plan exceeds this
    AIM_PERI = 150_000.0  # aim the approach at this target periapsis altitude
    # orbit-first arrival: capture right at the parking altitude (the moon
    # pilot's injection loop descends at <= 25 m/s, so a parking orbit far
    # below the capture would take hours to reach)
    CAPTURE_ALT = 230_000.0
    PARK_ALT = 150_000.0
    DEORBIT_PERI = -60_000.0
    DEORBIT_LEAD = 0.20
    REPLAN_EVERY = 600.0  # s between free-arrival-time re-plans
    LANDING_RESERVE = 900.0  # s kept before the time limit for the landing

    def __init__(self, env: ToMarsEnv):
        super().__init__(env, divert=False, orbit=False)
        self.v_inf_target = 0.0
        self.depart_dv = 0.0
        self.depart_dv0 = 0.0
        self.t_arrival = 0.0
        self._mcc_spent = 0.0
        self._cruise_dv0 = None
        self._last_plan = -1e9
        self._aim_offset = (0.0, 0.0)
        self.plan_log = []

    # the moon pilot's helpers read "the moon" through these two
    def _state(self):
        e = self.env
        r = np.array([e._rx, e._ry])
        v = np.array([e._vx, e._vy])
        ang = e._mars_angle()
        w = e.target_w
        mpos = np.array(e._mars_world())
        mvel = np.array(e._mars_velocity())
        return r, v, ang, w, mpos, mvel

    def _site_dir(self, tau: float) -> np.ndarray:
        a = self.env._mars_angle(self.env._time + tau) + self.env._site_angle
        return np.array([math.cos(a), math.sin(a)])

    # -- departure solver -------------------------------------------------------

    def _gap_from_state(self, x, y, vx, vy, t0: float, dv: float = 0.0):
        """Arrival phase gap of the heliocentric conic through (x, y, vx, vy)
        at time t0: the target's angle when the conic next crosses the
        target's orbit radius minus the conic's angle there (negative: the
        target is behind the arrival point). `dv` applies a prograde speed
        change first. Returns (gap, time to crossing) or None if the conic
        never reaches that radius."""
        e = self.env
        mu, r2 = e.star.mu, e.target.orbit_radius
        sp = math.hypot(vx, vy)
        if dv:
            vx, vy = vx * (1.0 + dv / sp), vy * (1.0 + dv / sp)
        rn = math.hypot(x, y)
        v2 = vx * vx + vy * vy
        rv = x * vx + y * vy
        h = x * vy - y * vx
        ex = ((v2 - mu / rn) * x - rv * vx) / mu
        ey = ((v2 - mu / rn) * y - rv * vy) / mu
        ecc = math.hypot(ex, ey)
        energy = 0.5 * v2 - mu / rn
        if energy >= 0.0 or ecc < 1e-6:
            return None
        a = -mu / (2.0 * energy)
        if a * (1.0 + ecc) < r2 or a * (1.0 - ecc) > r2:
            return None
        pp = h * h / mu
        nu_now = math.copysign(math.acos(max(-1.0, min(1.0, (pp / rn - 1.0) / ecc))), rv)
        nu2 = math.acos(max(-1.0, min(1.0, (pp / r2 - 1.0) / ecc)))  # outbound crossing

        def mean_anom(nu):
            E = 2.0 * math.atan2(
                math.sqrt(1.0 - ecc) * math.sin(nu / 2.0), math.sqrt(1.0 + ecc) * math.cos(nu / 2.0)
            )
            return E - ecc * math.sin(E)

        n = math.sqrt(mu / a**3)
        t2 = ((mean_anom(nu2) - mean_anom(nu_now)) % TWO_PI) / n
        sgn = 1.0 if h >= 0.0 else -1.0
        theta_arr = math.atan2(y, x) + sgn * ((nu2 - nu_now) % TWO_PI)
        return _wrap_pi(e._mars_angle(t0 + t2) - theta_arr), t2

    def _departure_solution(self, r, v):
        """Cheapest departure available from the rocket's CURRENT position:
        (v_inf, gap). For each hyperbolic excess speed the outgoing asymptote
        follows from where the rocket is (burn now, at periapsis of the
        escape hyperbola); the heliocentric conic from the SOI exit is
        propagated to the target's orbit and its phase gap read off. Any
        burn point is allowed -- an off-prograde asymptote just costs a
        little more excess speed -- so the window is continuous instead of
        once per parking orbit."""
        e, p = self.env, self.env.planet
        rn = float(np.linalg.norm(r))
        theta_r = math.atan2(r[1], r[0])
        v_h = math.sqrt(e.star.mu / e.home_r)
        v_inf_min = v_h * (
            math.sqrt(2.0 * e.target.orbit_radius / (e.home_r + e.target.orbit_radius)) - 1.0
        )
        best = None
        for k in range(60):
            v_inf = v_inf_min * (1.0 + 1.6 * k / 59.0)
            e_h = 1.0 + rn * v_inf * v_inf / p.mu
            nu_inf = math.acos(-1.0 / e_h)
            psi = theta_r + nu_inf  # CCW orbit: asymptote direction
            a_h = p.mu / (v_inf * v_inf)
            cosh_H = (1.0 + e.planet_soi / a_h) / e_h
            H = math.acosh(max(1.0, cosh_H))
            t_exit = (e_h * math.sinh(H) - H) / math.sqrt(p.mu / a_h**3)
            t_dep = e._time + t_exit
            hx, hy = e._home_helio(t_dep)
            hvx, hvy = e._home_helio_velocity(t_dep)
            x = hx + e.planet_soi * math.cos(psi)
            y = hy + e.planet_soi * math.sin(psi)
            res = self._gap_from_state(
                x, y, hvx + v_inf * math.cos(psi), hvy + v_inf * math.sin(psi), t_dep
            )
            if res is None:
                continue
            gap, t2 = res
            if abs(gap) < self.GAP_TOL:
                return v_inf, gap, t_dep + t2  # cheapest acceptable
            if best is None or abs(gap) < abs(best[1]):
                best = (v_inf, gap, t_dep + t2)
        return best

    def _plan_arrival(self, state=None, now=None):
        """Free-arrival-time Lambert targeting: scan arrival times between
        ten minutes from `now` and the latest the landing still fits, keep
        the one whose transfer needs the smallest burn from the given
        heliocentric state (default: the rocket's, now). Returns
        (t_arrival, dv_magnitude) or None."""
        e = self.env
        x, y, vx, vy = e.rocket_helio() if state is None else state
        now = e._time if now is None else now
        t_max = e.max_episode_steps * e.step_duration - self.LANDING_RESERVE
        if t_max - now < 600.0:
            return None
        best = None
        for k in range(48):
            t_arr = now + 600.0 + (t_max - now - 600.0) * k / 47.0
            sol = lambert(e.star.mu, (x, y), e._mars_helio(t_arr), t_arr - now)
            if sol is None:
                continue
            mag = math.hypot(sol[0][0] - vx, sol[0][1] - vy)
            if best is None or mag < best[1]:
                best = (t_arr, mag)
        return best

    def _tidal(self, rx: float, ry: float, t: float):
        """Star + target tidal acceleration in the planet frame at time t
        (the env's own formulation, with explicit time for look-ahead)."""
        e = self.env
        ax = ay = 0.0
        for (bx, by), mu in ((e._sun_world(t), e.star.mu), (e._mars_world(t), e.target.mu)):
            dx, dy = bx - rx, by - ry
            r2 = dx * dx + dy * dy
            inv = mu / (r2 * math.sqrt(r2))
            b2 = bx * bx + by * by
            invb = mu / (b2 * math.sqrt(b2))
            ax += dx * inv - bx * invb
            ay += dy * inv - by * invb
        return ax, ay

    def _propagate(self, rx, ry, vx, vy, t, horizon: float, dt: float = 2.0):
        """Coast in the planet frame under planet gravity + tides until
        `horizon` seconds pass or the rocket is 3 planet-SOI out."""
        mu, lim = self.env.planet.mu, 3.0 * self.env.planet_soi
        n = int(horizon / dt)
        for _ in range(n):
            r2 = rx * rx + ry * ry
            g = -mu / (r2 * math.sqrt(r2))
            tx, ty = self._tidal(rx, ry, t)
            vx += (g * rx + tx) * dt
            vy += (g * ry + ty) * dt
            rx += vx * dt
            ry += vy * dt
            t += dt
            if r2 > lim * lim:
                break
        return rx, ry, vx, vy, t

    def _plan_departure(self, r, v):
        """Simulate-ahead departure planning. Patched conics are useless
        here -- the star's tide does ~1 km/s of work during the climb-out --
        so for a grid of prograde burns applied NOW, coast the full force
        model out of the planet's influence, then price the rest of the trip
        with free-arrival-time Lambert from the state reached. Returns
        (burn_dv, total_dv, t_arrival) for the cheapest, or None."""
        e = self.env
        vmag = float(np.linalg.norm(v))
        vhat = v / vmag
        best = None
        for k in range(10):
            dv = 650.0 + 750.0 * k / 9.0
            vx0, vy0 = v[0] + dv * vhat[0], v[1] + dv * vhat[1]
            rx, ry, vx, vy, t = self._propagate(r[0], r[1], vx0, vy0, e._time, 6_000.0)
            hx, hy = e._home_helio(t)
            hvx, hvy = e._home_helio_velocity(t)
            plan = self._plan_arrival((rx + hx, ry + hy, vx + hvx, vy + hvy), t)
            if plan is None:
                continue
            total = dv + plan[1]
            if best is None or total < best[1]:
                best = (dv, total, plan[0])
        return best

    def _aim_point(self, t_arr: float):
        """Target position at arrival, offset sideways by the impact parameter
        that gives a periapsis at AIM_PERI for the approach velocity the
        Lambert transfer would arrive with (two-pass: solve to the centre,
        read the arrival velocity, offset, re-solve on the caller's side)."""
        e = self.env
        mx, my = e._mars_helio(t_arr)
        return mx + self._aim_offset[0], my + self._aim_offset[1]

    def _update_aim_offset(self, t_arr: float):
        e = self.env
        x, y, _vx, _vy = e.rocket_helio()
        t_go = t_arr - e._time
        sol = lambert(e.star.mu, (x, y), e._mars_helio(t_arr), t_go)
        if sol is None:
            return
        a = e._mars_angle(t_arr)
        sp = e.target_w * e.target.orbit_radius
        mvx, mvy = -sp * math.sin(a), sp * math.cos(a)
        wx, wy = sol[1][0] - mvx, sol[1][1] - mvy  # target-relative arrival velocity
        v_inf = max(50.0, math.hypot(wx, wy))
        rp = e.target.radius + self.AIM_PERI
        b = rp * math.sqrt(1.0 + 2.0 * e.target.mu / (rp * v_inf * v_inf))
        # offset perpendicular to the approach, on the side that makes the
        # flyby counter-clockwise about the target (the pilot's descent sense)
        ux, uy = wx / v_inf, wy / v_inf
        self._aim_offset = (b * uy, -b * ux)

    def _cruise_gap(self, dv: float = 0.0):
        x, y, vx, vy = self.env.rocket_helio()
        return self._gap_from_state(x, y, vx, vy, self.env._time, dv)

    # -- the mode machine -------------------------------------------------------

    def __call__(self, info: dict) -> np.ndarray:
        env, p = self.env, self.env.planet
        r, v, _, _, mpos, mvel = self._state()
        rn = float(np.linalg.norm(r))
        u = r / rn
        tw = np.array([-u[1], u[0]])  # CCW tangent: the parking orbit's sense
        vr = float(v @ u)
        vtw = float(v @ tw)
        alt, cutoff = rn - p.radius, p.space_altitude
        apo, peri = info["apoapsis"], info["periapsis"]
        prograde = math.atan2(vr, max(1.0, vtw))

        if self.mode == "ascend":
            if math.isfinite(apo) and apo >= self.h_ref:
                self.mode = "coast"
            elif alt < 250.0:
                return self._act_elev(math.pi / 2.0, 1.0, u, tw)
            else:
                f = min(1.0, (alt - 250.0) / (self.turn_end - 250.0))
                elev = math.radians(88.0) * (1.0 - f) ** self.TURN_EXP
                if vr < 40.0 and alt > 0.15 * cutoff:
                    elev = max(elev, math.radians(30.0))
                return self._act_elev(elev, 1.0, u, tw)
        if self.mode == "coast":
            if apo < 1.005 * cutoff:
                self.mode = "ascend"
                return self._act_elev(max(prograde, 0.0), 1.0, u, tw)
            if alt > apo - 2_000.0 or vr < 20.0:
                self.mode = "inject"
            else:
                return self._act_elev(prograde, 0.0, u, tw)
        if self.mode == "inject":
            rp = p.radius + self.peri_target
            v_need = math.sqrt(2.0 * p.mu * rp / (rn * (rn + rp)))
            if peri >= self.peri_target - 150.0:
                self.mode = "hold"
            else:
                a_av = self._accel_avail(info)
                g_eff = p.mu / (rn * rn) - vtw * vtw / rn
                vr_des = min(40.0, max(-15.0, 0.02 * (self.h_ref - alt)))
                a_v = g_eff + 0.4 * (vr_des - vr)
                a_h = min(a_av, max(0.0, 0.8 * (v_need - vtw)))
                return self._act_elev(
                    math.atan2(a_v, a_h), min(1.0, math.hypot(a_v, a_h) / a_av), u, tw
                )

        if self.mode == "hold":
            if info["orbit_progress"] >= 1.0 and env._steps % 20 == 0:
                plan = self._plan_departure(r, v)
                if plan is not None and plan[1] < self.DEPART_MAX:
                    self.depart_dv = plan[0]
                    self.depart_dv0 = env._dv_used
                    self.t_arrival = plan[2]
                    self.plan_log.append((env._time, plan[2], plan[1]))
                    self.mode = "depart"
            if self.mode != "depart":
                if peri < 1.015 * cutoff:
                    self._trim = True
                if self._trim and peri >= 1.025 * cutoff:
                    self._trim = False
                return self._act_elev(prograde, 0.3 if self._trim else 0.0, u, tw, gate=0.2)

        if self.mode == "depart":
            # cut off on delta-v delivered, not on speed: over a 100 s burn
            # gravity takes speed back and a speed target overshoots badly
            v_go = self.depart_dv - (env._dv_used - self.depart_dv0)
            if v_go <= 0.0:
                self.mode = "cruise"
            else:
                return self._act_world(
                    math.atan2(v[1], v[0]), min(1.0, max(0.05, v_go / 25.0)), gate=0.3
                )

        prel = r - mpos
        vrel = v - mvel
        d = float(np.linalg.norm(prel))

        if self.mode == "cruise":
            if d < 3.0 * env.mars_soi:
                self.mode = "capture"
            else:
                clear = rn > 2.5 * env.planet_soi
                if clear and env._time - self._last_plan > self.REPLAN_EVERY:
                    # Re-plan, but only switch arrival time when it pays: the
                    # free-time scan hops between minima and each hop costs a
                    # new residual (measured: 1.8 km/s of corrections wasted).
                    self._last_plan = env._time
                    plan = self._plan_arrival()
                    cur = None
                    if self.t_arrival > env._time + 300.0:
                        x, y, vx, vy = env.rocket_helio()
                        sol = lambert(
                            env.star.mu,
                            (x, y),
                            self._aim_point(self.t_arrival),
                            self.t_arrival - env._time,
                        )
                        if sol is not None:
                            cur = math.hypot(sol[0][0] - vx, sol[0][1] - vy)
                    if plan is not None and (
                        cur is None
                        or cur > self.REPLAN_RESIDUAL
                        or plan[1] < cur - self.REPLAN_GAIN
                    ):
                        self.t_arrival = plan[0]
                        self._update_aim_offset(plan[0])
                        self.plan_log.append((env._time, plan[0], plan[1]))
                t_go = self.t_arrival - env._time
                if self._cruise_dv0 is None:
                    self._cruise_dv0 = env._dv_used
                self._mcc_spent = env._dv_used - self._cruise_dv0  # actual, not commanded
                if clear and t_go > 300.0 and self._mcc_spent < self.MCC_BUDGET:
                    x, y, vx, vy = env.rocket_helio()
                    sol = lambert(env.star.mu, (x, y), self._aim_point(self.t_arrival), t_go)
                    if sol is not None:
                        dvx, dvy = sol[0][0] - vx, sol[0][1] - vy
                        mag = math.hypot(dvx, dvy)
                        if mag > self.MCC_MIN:
                            a_av = self._accel_avail_moon()
                            return self._act_world(
                                math.atan2(dvy, dvx),
                                min(1.0, mag / (a_av * env.step_duration)),
                                gate=0.3,
                            )
                return self._act_world(math.atan2(-vrel[1], -vrel[0]), 0.0)

        if self.mode in ("capture", "moon_orbit", "deorbit", "pdescend"):
            if self.mode == "capture":
                peri, _, energy, ecc = env._mars_orbit_elements()
                malt = d - env.target.radius
                # the moon pilot waits to drop below CAPTURE_ALT; if the aim was
                # off and the periapsis is higher, insert anyway once receding
                if energy >= 0.0 and malt > self.CAPTURE_ALT and float(prel @ vrel) > 0.0:
                    return self._act_world(math.atan2(-vrel[1], -vrel[0]), 1.0, gate=0.4)
                # bound and round: that is a parking orbit, hand over
                if energy < 0.0 and ecc < 0.08 and peri > 0.5 * self.PARK_ALT:
                    self.mode = "moon_orbit"
            return self._orbit_control(prel, vrel, d)

        if self.mode == "brake":
            malt = d - env.target.radius
            speed = float(np.linalg.norm(vrel))
            div = self._divert(prel, vrel, d)
            if div is not None:
                return div
            if (speed < 70.0 and malt < 8.0 * self.GATE_ALT) or malt < 1.2 * self.GATE_ALT:
                self.mode = "descend"
            else:
                return self._brake(info, prel, vrel, d)

        return self._descend(info, prel, vrel, d)
