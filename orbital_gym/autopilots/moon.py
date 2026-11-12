"""Scripted pilot for ToMoon: westward ascent, parking orbit, Kepler-timed
trans-lunar injection, and either a direct retro suicide burn or (orbit=True)
capture into a low lunar orbit, a timed deorbit and a powered descent."""

from __future__ import annotations

import math

import numpy as np

from ..missions.to_moon import ToMoonEnv
from ..missions.to_orbit import TWO_PI, _wrap_pi
from .base import throttle_action


class MoonAutopilot:
    """ascend -> coast -> inject -> hold -> tli -> transfer -> brake -> descend.

    Everything is flown westward: the moon orbits counterclockwise, and only
    a co-rotating approach keeps the excess arrival speed (~0.4 km/s over
    the moon's own escape velocity) affordable.
    The ascent is leaner than the base Autopilot's (~4.07 vs ~4.27 km/s):
    gravity-turn schedule, coast to apoapsis, then a level vacuum burn that
    holds altitude with gravity-feedforward attitude until the periapsis
    clears. TLI timing re-solves the phasing every step (Kepler time to the
    moon-orbit crossing) and fires when the required transfer apoapsis
    enters a narrow band just past the moon's orbit -- robust to upstream
    timing shifts. Arrival brakes retrograde as late as the stopping
    distance allows (with a vertical-support tilt once below moon-circular
    speed), then descends on the suicide-burn boundary, tracking the site
    only when it is within cheap reach.

    Tuned on the "easy" planet/moon pair; the delta-v budget does not close
    on "earth". On "easy" it touches down at the descent profile's ~8 m/s
    floor with ~3.4 km/s to spare -- but a couple hundred km off the site.
    Reaching the site is what train_guided.py leaves for RL to learn.
    """

    PERI_FRAC = 1.03  # parking-orbit periapsis, * atmosphere cutoff
    LEVEL_FRAC = 1.04  # level-burn altitude, * cutoff (must exceed PERI_FRAC)
    TURN_END_FRAC = 0.5  # pitch schedule reaches horizontal at this * cutoff
    TURN_EXP = 0.78
    TLI_BAND = (0.995, 1.005)  # accepted transfer apoapsis, * moon orbit radius
    BRAKE_MARGIN = 0.93  # accel fraction the stopping-distance trigger assumes
    BRAKE_LEAD = 1.28  # stopping distance vs radial gap; center of the
    # landing plateau [1.20, 1.35] on the moon rocket
    # (scanned; the Keplerian moon's arrival is slow
    # and shallow, so the flight path runs well past
    # the radial gap)
    GATE_ALT = 4_000.0  # descend handoff altitude scale, m
    SITE_REACH = 30_000.0  # only steer for the site when within this, m
    # -- site targeting (the "divert"). The touchdown longitude is the
    # periapsis of the moon-relative approach, fixed by the transfer, so the
    # terminal descent's ~30 km fine-steering cannot cover the ~290 km the raw
    # approach misses the site by. The fix is to aim the approach itself, high
    # and far out where a small cross-track burn has a long lever arm and there
    # is no retro burn to steal thrust from; below HANDOFF_ALT the normal
    # brake+descend lands, now with the site inside its reach.
    HANDOFF_ALT = 120_000.0  # aim above this, brake/land below, m
    DIVERT_N = 3.0  # ZEM cross-track navigation gain
    DIVERT_CAP = 0.4  # max fraction of thrust spent aiming
    DIVERT_DEADBAND = 4_000.0  # stop aiming once predicted miss is under this, m
    DIVERT_BUDGET = 2_500.0  # delta-v ceiling on aiming, m/s: past it, stop and
    # land soft where we are rather than run the tank
    # dry braking (a soft off-site landing beats a crash)
    # -- orbit-first capture (opt-in). Rather than a direct suicide descent, the
    # vehicle brakes into a low circular parking orbit; from there the landing
    # site is reachable by choosing WHEN to deorbit (KSP-style), which turns
    # site targeting from a thrust-saturated cross-track fight into a clean
    # timing decision -- the shape the RL problem needs.
    CAPTURE_ALT = 140_000.0  # begin the capture burn at this moon altitude, m
    PARK_ALT = 60_000.0  # target circular parking-orbit altitude, m
    DEORBIT_LEAD = 0.20  # deorbit when the site is this far ahead in the
    # orbit (rad); the descent then lands on it. Measured
    # constant across sites (one value hits -90..+150 deg
    # on-site) -- it is a property of the descent
    # geometry, so the SAME wait hits any site.
    DEORBIT_PERI = -40_000.0  # burn retrograde until periapsis drops to here, m

    def __init__(self, env: ToMoonEnv, divert: bool = False, orbit: bool = False):
        self.env = env
        # Orbit-first is opt-in: off, the pilot flies the frozen direct-descent
        # baseline. On, it captures into a low circular moon orbit before the
        # descent -- the setup the from-orbit RL landing trains against.
        self.orbit = orbit
        # Site targeting is opt-in: off, the pilot is the frozen soft-landing
        # baseline every test, train_guided cache, and default flight relies on
        # (lands ~290 km off site at 8 m/s). On, it aims the approach at the
        # env's site during the coast. It reaches sites within a cone of the
        # natural landing; far sites (e.g. the preset 0 deg) are out of reach
        # and it lands soft where it can.
        self.enable_divert = divert
        self.mode = "ascend"
        p = env.planet
        self.peri_target = self.PERI_FRAC * p.space_altitude
        self.h_ref = self.LEVEL_FRAC * p.space_altitude
        self.turn_end = self.TURN_END_FRAC * p.space_altitude
        self._ra_target = 0.0
        self._burning = False
        self._trim = False
        self._dv_at_divert = None  # delta-v when aiming began (budget tracking)

    # -- state helpers (world frame; the moon is on rails, so its state is
    # a closed-form function of time)

    def _state(self):
        e, moon = self.env, self.env.moon
        r = np.array([e._rx, e._ry])
        v = np.array([e._vx, e._vy])
        ang = TWO_PI * (e._time / moon.orbit_period)
        w = TWO_PI / moon.orbit_period
        mpos = moon.orbit_radius * np.array([math.cos(ang), math.sin(ang)])
        mvel = w * moon.orbit_radius * np.array([-math.sin(ang), math.cos(ang)])
        return r, v, ang, w, mpos, mvel

    def _site_dir(self, tau: float) -> np.ndarray:
        moon = self.env.moon
        a = TWO_PI * ((self.env._time + tau) / moon.orbit_period) + moon.site_angle
        return np.array([math.cos(a), math.sin(a)])

    def _accel_avail(self, info: dict) -> float:
        return self.env.rocket.stages[self.env._stage].thrust / info["mass"]

    def _act_world(self, theta: float, throttle: float, gate: float | None = None):
        e = self.env
        err = _wrap_pi(theta - e._theta)
        if gate is not None and abs(err) > gate:
            throttle = 0.0  # don't thrust mid-slew
        if e.attitude_control:
            # project the desired world direction onto the local frame
            ux, uy, tx, ty = e._local_frame()
            wx, wy = math.cos(theta), math.sin(theta)
            return np.array(
                [throttle_action(self.env, throttle), wx * ux + wy * uy, wx * tx + wy * ty],
                dtype=np.float32,
            )
        turn = np.clip(err / (e.rocket.max_turn_rate * e.step_duration), -1.0, 1.0)
        return np.array([throttle_action(self.env, throttle), turn], dtype=np.float32)

    def _act_elev(self, elev: float, throttle: float, u, tw, gate=None):
        d = math.cos(elev) * tw + math.sin(elev) * u
        return self._act_world(math.atan2(d[1], d[0]), throttle, gate)

    def __call__(self, info: dict) -> np.ndarray:
        env, p, moon = self.env, self.env.planet, self.env.moon
        r, v, moon_ang, w_m, mpos, mvel = self._state()
        rn = float(np.linalg.norm(r))
        u = r / rn
        tw = np.array([-u[1], u[0]])  # "west" tangent: the moon's direction
        vr = float(v @ u)
        vtw = float(v @ tw)
        alt, cutoff = rn - p.radius, p.space_altitude
        apo, peri = info["apoapsis"], info["periapsis"]
        prograde = math.atan2(vr, max(1.0, vtw))

        # -- ascent: gravity-turn schedule, coast up, level injection burn
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
            if apo < 1.005 * cutoff:  # drag ate the apoapsis: climb again
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
                # command an acceleration vector: vertical holds altitude
                # (gravity feedforward + vr loop), horizontal tapers into
                # the cutoff. Attitude and throttle both come from the
                # vector, so the vertical hold survives the throttle-down.
                a_av = self._accel_avail(info)
                g_eff = p.mu / (rn * rn) - vtw * vtw / rn
                vr_des = min(40.0, max(-15.0, 0.02 * (self.h_ref - alt)))
                a_v = g_eff + 0.4 * (vr_des - vr)
                a_h = min(a_av, max(0.0, 0.8 * (v_need - vtw)))
                return self._act_elev(
                    math.atan2(a_v, a_h), min(1.0, math.hypot(a_v, a_h) / a_av), u, tw
                )

        # -- hold: coast until the orbit-hold is complete AND a transfer
        # solution exists. Each step, solve (Kepler) for the apoapsis a
        # TLI-right-now would need to meet the moon at the orbit crossing;
        # fire when it enters the band. No instantaneous window to miss.
        if self.mode == "hold":
            if info["orbit_progress"] >= 1.0:
                r2 = moon.orbit_radius
                theta_r = math.atan2(r[1], r[0])

                def phase_gap(ra: float) -> float:
                    a_sma = 0.5 * (rn + ra)
                    e = (ra - rn) / (ra + rn)
                    r_t = min(r2, ra * 0.999999)
                    cosnu = (a_sma * (1.0 - e * e) / r_t - 1.0) / max(e, 1e-9)
                    nu = math.acos(max(-1.0, min(1.0, cosnu)))
                    ecc_an = 2.0 * math.atan2(
                        math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                        math.sqrt(1.0 + e) * math.cos(nu / 2.0),
                    )
                    t_x = (ecc_an - e * math.sin(ecc_an)) / math.sqrt(p.mu / a_sma**3)
                    return _wrap_pi(moon_ang + w_m * t_x - theta_r - nu)

                lo_f, hi_f = self.TLI_BAND
                best_ra, best_gap = 0.0, 0.02
                for k in range(40):
                    ra = (lo_f + k * (hi_f - lo_f) / 39.0) * r2
                    g = abs(phase_gap(ra))
                    if g < best_gap:
                        best_ra, best_gap = ra, g
                if best_ra > 0.0:
                    self._ra_target = best_ra
                    self.mode = "tli"
            if self.mode != "tli":
                # station-keeping: the moon's pull can sag the periapsis
                # into the atmosphere over a long phase wait
                if peri < 1.015 * cutoff:
                    self._trim = True
                if self._trim and peri >= 1.025 * cutoff:
                    self._trim = False
                return self._act_elev(prograde, 0.3 if self._trim else 0.0, u, tw, gate=0.2)

        if self.mode == "tli":
            # burn prograde up to the transfer-ellipse speed, throttle
            # tapered (apoapsis is far too cutoff-sensitive for 1 s steps)
            v_go = math.sqrt(p.mu * (2.0 / rn - 2.0 / (rn + self._ra_target))) - math.hypot(
                float(env._vx), float(env._vy)
            )
            if v_go <= 0.0:
                self.mode = "transfer"
            else:
                return self._act_elev(prograde, min(1.0, max(0.05, v_go / 25.0)), u, tw, gate=0.3)

        prel = r - mpos
        vrel = v - mvel
        d = float(np.linalg.norm(prel))

        if self.mode == "transfer":
            if d < 6.0 * moon.radius:
                self.mode = "capture" if self.orbit else "brake"
            else:
                div = self._divert(prel, vrel, d)
                return div if div is not None else self._act_elev(prograde, 0.0, u, tw)

        if self.mode in ("capture", "moon_orbit", "deorbit", "pdescend"):
            return self._orbit_control(prel, vrel, d)

        if self.mode == "brake":
            malt = d - moon.radius
            speed = float(np.linalg.norm(vrel))
            div = self._divert(prel, vrel, d)
            if div is not None:
                return div
            if (speed < 70.0 and malt < 8.0 * self.GATE_ALT) or malt < 1.2 * self.GATE_ALT:
                self.mode = "descend"
            else:
                return self._brake(info, prel, vrel, d)

        return self._descend(info, prel, vrel, d)

    def _orbit_control(self, prel, vrel, d: float) -> np.ndarray:
        """Capture into, then hold, a low circular moon orbit.

        On the incoming hyperbolic approach the vehicle coasts (pointed
        retrograde, ready) until it drops to CAPTURE_ALT, then runs an
        injection controller in the moon frame: null the radial velocity while
        driving the tangential speed to circular at PARK_ALT. Centripetal and
        gravity terms are fed forward so the throttle-down doesn't drop the
        altitude. Once radial velocity is killed and the speed is circular with
        periapsis clear of the surface, it switches to a coasting hold -- the
        parking orbit the RL landing takes over from.
        """
        e = self.env
        moon = e.moon
        um = prel / d
        tm = np.array([-um[1], um[0]])  # +90 deg: CCW tangential
        v_r = float(vrel @ um)
        v_t = float(vrel @ tm)
        sgn = 1.0 if v_t >= 0.0 else -1.0  # keep the approach's orbital sense
        malt = d - moon.radius
        g_m = moon.mu / (d * d)
        v_circ = math.sqrt(moon.mu / d)
        a_av = self._accel_avail_moon()
        peri, _apo, _energy, ecc = e._moon_orbit_elements()

        if self.mode == "capture":
            # coast in retrograde until low enough to start the capture burn
            if malt > self.CAPTURE_ALT and v_r < 0.0:
                return self._act_world(math.atan2(-vrel[1], -vrel[0]), 0.0)
            if abs(v_r) < 12.0 and abs(v_t - sgn * v_circ) < 20.0 and peri > 0.0:
                self.mode = "moon_orbit"

        if self.mode == "moon_orbit":
            # deorbit when the site sits the right lead-angle ahead in the orbit
            theta_v = math.atan2(prel[1], prel[0])
            site_world = e._moon_angle() + e._site_angle
            ahead = sgn * _wrap_pi(site_world - theta_v)  # +: site ahead of us
            if 0.0 < ahead <= self.DEORBIT_LEAD:
                self.mode = "deorbit"
            elif peri > 0.5 * self.PARK_ALT and ecc < 0.08:
                # hold: only trim if the orbit has decayed (tide, sloppy capture)
                return self._act_world(math.atan2(-vrel[1], -vrel[0]), 0.0)

        if self.mode == "deorbit":
            if peri > self.DEORBIT_PERI:  # burn retrograde to drop periapsis
                return self._act_world(math.atan2(-vrel[1], -vrel[0]), 1.0, gate=0.4)
            self.mode = "pdescend"  # periapsis below surface: powered descent

        if self.mode == "pdescend":
            return self._pdescend(prel, vrel, d)

        # injection burn: v_r -> a gentle altitude-hold target, v_t -> circular
        vr_des = max(-25.0, min(25.0, 0.05 * (self.PARK_ALT - malt)))
        a_r = g_m - v_t * v_t / d + 0.6 * (vr_des - v_r)  # gravity + centripetal
        a_t = 0.7 * (sgn * v_circ - v_t)
        a_cmd = a_r * um + a_t * tm
        mag = float(np.linalg.norm(a_cmd))
        return self._act_world(math.atan2(a_cmd[1], a_cmd[0]), min(1.0, mag / a_av), gate=0.6)

    def _pdescend(self, prel, vrel, d: float) -> np.ndarray:
        """Powered descent from the deorbit, landing soft.

        The hyperbolic-approach brake/descend overshoots here -- it nearly
        stops the vehicle kilometres up, then hovers the tank dry. From orbit
        the right law is one continuous braking arc: ride a suicide-burn
        profile on TOTAL moon-relative speed (v <= sqrt(2*a_net*altitude)),
        thrust retrograde so the same burn kills the ~1 km/s of horizontal
        velocity and the descent together, and blend in radial support near the
        ground so the last hundred metres are gentle rather than a dive.
        """
        m = self.env.moon
        um = prel / d
        tm = np.array([-um[1], um[0]])
        malt = d - m.radius
        v = float(np.linalg.norm(vrel))
        vhat = vrel / max(1.0, v)
        a_av = self._accel_avail_moon()
        g_m = m.mu / (d * d)
        v_r = float(vrel @ um)  # radial velocity (negative descending)
        v_t = float(vrel @ tm)

        # max speed we can still bleed to zero by the surface, with margin
        a_net = max(0.5, 0.85 * a_av - g_m)
        v_prof = math.sqrt(2.0 * a_net * max(0.0, malt))

        if malt < 150.0:
            # terminal: null horizontal, settle down at a few m/s
            vr_des = -min(3.0, 0.05 * malt + 1.0)
            a_r = g_m + 0.8 * (vr_des - v_r)
            a_t = -0.8 * v_t
            a_cmd = a_r * um + a_t * tm
            return self._act_world(
                math.atan2(a_cmd[1], a_cmd[0]),
                min(1.0, float(np.linalg.norm(a_cmd)) / a_av),
                gate=0.6,
            )

        if v > 0.9 * v_prof:
            # on/above the profile: brake retrograde at full thrust, with just
            # enough radial support that the vertical rate can't run away
            support = max(0.0, g_m - v * v / d)
            retro = math.sqrt(max(1.0, a_av * a_av - support * support))
            a_cmd = -vhat * retro + um * support
            return self._act_world(math.atan2(a_cmd[1], a_cmd[0]), 1.0, gate=0.5)

        # under the profile: coast down at a controlled sink, holding attitude
        vr_des = -max(5.0, 0.5 * v_prof)
        a_r = max(0.0, g_m + 0.6 * (vr_des - v_r))
        a_t = -0.6 * v_t
        a_cmd = a_r * um + a_t * tm
        return self._act_world(
            math.atan2(a_cmd[1], a_cmd[0]), min(1.0, float(np.linalg.norm(a_cmd)) / a_av), gate=0.6
        )

    def _divert(self, prel, vrel, d: float):
        """Aim the coasting approach at the site with ZEM cross-track guidance.

        Returns an action while a correction is worth making, or None to fall
        through to the normal coast/brake. Zero-effort-miss: project where the
        approach will be at closest approach, compare to where the site will
        be, and burn the cross-track component away. It self-tapers -- as the
        predicted miss shrinks the burn shrinks -- and stops entirely inside
        the deadband, so it costs only the delta-v the aim actually needs.
        """
        e = self.env
        if not self.enable_divert or self._burning or (d - e.moon.radius) < self.HANDOFF_ALT:
            return None
        dv_now = e.delta_v_remaining()
        if self._dv_at_divert is None:
            self._dv_at_divert = dv_now
        if self._dv_at_divert - dv_now > self.DIVERT_BUDGET:
            return None  # spent the aiming budget: keep the rest for landing
        pv = float(prel @ vrel)
        if pv >= 0.0:  # past closest approach / not approaching: nothing to aim
            return None
        vv = max(1.0, float(vrel @ vrel))
        t_go = max(2.0, -pv / vv)
        vhat = vrel / math.sqrt(vv)
        mpos = np.array(e._moon_world())
        site_rel = np.array(e._moon_site_world()) - mpos
        zem = site_rel - (prel + vrel * t_go)
        zem_lat = zem - float(zem @ vhat) * vhat
        if float(np.linalg.norm(zem_lat)) < self.DIVERT_DEADBAND:
            return None
        a_av = self._accel_avail_moon()
        a_lat = self.DIVERT_N * zem_lat / (t_go * t_go)
        mag = float(np.linalg.norm(a_lat))
        if mag > self.DIVERT_CAP * a_av:
            a_lat *= self.DIVERT_CAP * a_av / mag
            mag = self.DIVERT_CAP * a_av
        return self._act_world(math.atan2(a_lat[1], a_lat[0]), min(1.0, mag / a_av), gate=0.6)

    def _accel_avail_moon(self) -> float:
        return self.env.rocket.stages[self.env._stage].thrust / self.env._mass

    def _brake(self, info: dict, prel, vrel, d: float) -> np.ndarray:
        """Moon-relative retro suicide burn, as late as possible.

        Bang-bang at full thrust minimizes the time gravity has to feed
        speed back in; a small lateral term steers for the site only when
        the predicted miss is already cheap to fix.
        """
        moon = self.env.moon
        v = float(np.linalg.norm(vrel))
        vhat = vrel / max(1.0, v)
        a_av = self._accel_avail(info)
        g_m = moon.mu / (d * d)

        # straight-line time to impact (or to closest approach if missing)
        pv = float(prel @ vrel)
        vv = max(1.0, float(vrel @ vrel))
        disc = pv * pv - vv * (float(prel @ prel) - moon.radius**2)
        t_go = max(2.0, (-pv - math.sqrt(disc)) / vv if disc > 0.0 else -pv / vv)

        a_lat = np.zeros(2)
        if t_go < 150.0:
            zem = prel + vrel * t_go - self._site_dir(t_go) * moon.radius
            zem_lat = zem - float(zem @ vhat) * vhat
            if float(np.linalg.norm(zem_lat)) < self.SITE_REACH:
                a_lat = 4.0 * zem_lat / (t_go * t_go)
                lat_mag = float(np.linalg.norm(a_lat))
                if lat_mag > 0.15 * a_av:
                    a_lat *= 0.15 * a_av / lat_mag

        a_net = max(1.0, self.BRAKE_MARGIN * a_av - g_m)
        if pv < 0.0 and v * v / (2.0 * a_net) >= self.BRAKE_LEAD * max(1.0, d - moon.radius):
            self._burning = True
        if pv >= 0.0 and v > 300.0:
            self._burning = False  # never chase a missed flyby
        if not self._burning:
            # coast pointing retrograde so the burn starts aligned
            return self._act_world(math.atan2(-vrel[1], -vrel[0]), 0.0)
        # below moon-circular speed gravity outruns the centrifugal term:
        # tilt enough thrust inward-out to keep the path from steepening
        # (a reverse gravity turn), spend the rest retrograde
        support = max(0.0, g_m - v * v / d)
        retro = math.sqrt(max(1.0, a_av * a_av - support * support))
        a_cmd = -vhat * retro + (prel / d) * support + a_lat
        return self._act_world(math.atan2(a_cmd[1], a_cmd[0]), 1.0, gate=0.5)

    def _descend(self, info: dict, prel, vrel, d: float) -> np.ndarray:
        """Vertical descent riding the suicide-burn boundary (fall free,
        brake late), tracking the site laterally only when it is close."""
        moon = self.env.moon
        um = prel / d
        tm = np.array([-um[1], um[0]])
        malt = d - moon.radius
        g_m = moon.mu / (d * d)
        vr_rel = float(vrel @ um)
        vt_rel = float(vrel @ tm)
        tau = min(120.0, malt / max(8.0, -vr_rel))
        site = self._site_dir(tau)
        x_err = _wrap_pi(math.atan2(site[1], site[0]) - math.atan2(prel[1], prel[0])) * d
        if abs(x_err) > self.SITE_REACH:
            x_err = 0.0  # site out of cheap reach: land soft where we are
        vt_des = max(-60.0, min(60.0, 0.05 * x_err))
        slow = 1.0 / (1.0 + abs(x_err) / 2_000.0)
        a_av = self._accel_avail(info)
        # ride well inside the suicide boundary (v^2 = 2(a-g)d): the profile's
        # required thrust is k/2 + g_m, and tracking it with a P-loop needs
        # headroom -- k near the boundary commands an uncatchable dive
        k = 0.8 * max(2.0, a_av - g_m)
        # floor well under the 25 m/s soft-landing threshold: the touchdown
        # speed is this floor plus tracking residue
        vr_prof = slow * math.sqrt(k * max(0.0, malt - 25.0))
        vr_des = -max(8.0, vr_prof)
        # riding v = s*sqrt(k*d) takes s^2*k/2 of deceleration on top of
        # gravity; without this feedforward the P-loop carries a standing
        # (k/2)/gain speed error all the way to the ground, which is the
        # difference between touching down at ~10 m/s and at ~25. Only while
        # actually on the profile from above: while still under it the right
        # move is free fall, and feeding the decel forward there means
        # hovering down on the engine until the tank runs dry
        ff = 0.5 * k * slow * slow if (vr_prof > 8.0 and vr_rel <= vr_des) else 0.0
        # never thrust downward: above the boundary profile the right move
        # is free fall, not a powered dive that must itself be killed
        a_r = max(0.0, g_m + ff + 0.8 * (vr_des - vr_rel))
        a_cmd = a_r * um + 0.8 * (vt_des - vt_rel) * tm
        return self._act_world(
            math.atan2(a_cmd[1], a_cmd[0]), min(1.0, float(np.linalg.norm(a_cmd)) / a_av)
        )
