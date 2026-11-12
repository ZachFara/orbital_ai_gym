"""Scripted autopilots: proof that the missions are solvable.

Ascent flies a classic gravity turn: vertical rise, pitch-over schedule down
to the horizon, cut the engine once apoapsis is high enough, coast, then burn
prograde near apoapsis until the periapsis is out of the atmosphere. The
return mission continues: hold orbit for the required half revolution, burn
retrograde at a fixed lead angle before the pad, fall through the atmosphere
(drag does most of the braking), and finish with a vertical landing burn.

The moon mission launches WESTWARD (the moon orbits counterclockwise, so an
eastward orbit meets it head-on at ~2 km/s closing speed), holds a low
parking orbit, fires trans-lunar injection when the phasing solution enters
its window, coasts out, and brakes retrograde into a soft off-site landing.
Putting the touchdown ON the site is deliberately left to RL
(train_guided.py starts its policies from this pilot's flights).

Run:  python scripts/autopilot_demo.py [--mission orbit|return|moon] [--planet easy|earth] [--render]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import orbital_gym
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_mars import ToMarsEnv
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.missions.to_orbit import TWO_PI, TwoStageToOrbitEnv, _wrap_pi


def _throttle_action(env, throttle: float) -> float:
    """Map a desired throttle to the a[0] the env expects. With hover_control
    the action is hover-relative (throttle = hover * exp(a[0]*K)), so invert
    that; otherwise it is the plain [0,1] -> [-1,1]. Shared by every autopilot."""
    if getattr(env, "hover_control", False):
        hover = max(1e-4, env.hover_throttle())
        a0 = math.log(max(1e-4, throttle) / hover) / env.HOVER_K
        return float(np.clip(a0, -1.0, 1.0))
    return 2.0 * throttle - 1.0


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
                [_throttle_action(self.env, throttle), math.sin(pitch), math.cos(pitch)],
                dtype=np.float32,
            )
        err = _wrap_pi(pitch - math.radians(info["pitch_deg"]))
        turn = np.clip(err / (env.rocket.max_turn_rate * env.step_duration), -1.0, 1.0)
        return np.array([_throttle_action(self.env, throttle), turn], dtype=np.float32)


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
                [_throttle_action(self.env, throttle), wx * ux + wy * uy, wx * tx + wy * ty],
                dtype=np.float32,
            )
        turn = np.clip(err / (e.rocket.max_turn_rate * e.step_duration), -1.0, 1.0)
        return np.array([_throttle_action(self.env, throttle), turn], dtype=np.float32)

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


def lambert(mu: float, r1v, r2v, dt: float, prograde: bool = True):
    """Two-body Lambert problem (universal variables, bisection on z):
    the velocity at r1v that reaches r2v after dt. Returns (v1, v2) as
    numpy arrays, or None if no solution converged."""
    r1v = np.asarray(r1v, dtype=float)
    r2v = np.asarray(r2v, dtype=float)
    r1, r2 = float(np.linalg.norm(r1v)), float(np.linalg.norm(r2v))
    cross = r1v[0] * r2v[1] - r1v[1] * r2v[0]
    dth = math.acos(max(-1.0, min(1.0, float(r1v @ r2v) / (r1 * r2))))
    if (prograde and cross < 0.0) or (not prograde and cross >= 0.0):
        dth = TWO_PI - dth
    A = math.sin(dth) * math.sqrt(r1 * r2 / max(1e-12, 1.0 - math.cos(dth)))
    if abs(A) < 1e-9:
        return None

    def stumpff(z):
        if z > 1e-8:
            sz = math.sqrt(z)
            return (1.0 - math.cos(sz)) / z, (sz - math.sin(sz)) / (sz**3)
        if z < -1e-8:
            sz = math.sqrt(-z)
            return (math.cosh(sz) - 1.0) / (-z), (math.sinh(sz) - sz) / (sz**3)
        return 0.5, 1.0 / 6.0

    def y_of(z):
        C, S = stumpff(z)
        return r1 + r2 + A * (z * S - 1.0) / math.sqrt(C), C, S

    def F(z):
        y, C, S = y_of(z)
        if y < 0.0:
            return None
        return (y / C) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt

    lo, hi = -4.0 * math.pi**2, 4.0 * math.pi**2 - 1e-6
    f_lo = F(lo)
    while f_lo is None and lo < hi:  # y < 0: raise the lower bracket
        lo += 1.0
        f_lo = F(lo)
    f_hi = F(hi)
    if f_lo is None or f_hi is None or f_lo > 0.0 or f_hi < 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = F(mid)
        if f_mid is None or f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    y, _C, _S = y_of(0.5 * (lo + hi))
    f = 1.0 - y / r1
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2
    v1 = (r2v - f * r1v) / g
    v2 = (gdot * r2v - r1v) / g
    return v1, v2


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


def fly(env: TwoStageToOrbitEnv, verbose: bool = True, pilot=None):
    _obs, info = env.reset()
    pilot = pilot or Autopilot(env)
    total_reward, last_print = 0.0, -1e9
    while True:
        action = pilot(info)
        _obs, reward, terminated, truncated, info = env.step(action)
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
    ap.add_argument("--mission", default="orbit", choices=["orbit", "return", "moon", "mars"])
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
    elif args.mission == "moon":
        env = ToMoonEnv(**kwargs)
        pilot = MoonAutopilot(env)
    elif args.mission == "mars":
        kwargs.pop("planet", None)
        env = ToMarsEnv(**kwargs)
        pilot = MarsAutopilot(env)
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
    print(
        f"orbit: {info['periapsis'] / 1000:.1f} x {apo_s} (cutoff {env.planet.space_altitude / 1000:.0f} km)"
    )
    if "pad_downrange" in info:
        print(
            f"touchdown {info['pad_downrange'] / 1000:.2f} km from the pad "
            f"at {info['speed']:.1f} m/s"
        )
    if "mars_distance" in info:
        print(
            f"mars: {info['mars_altitude'] / 1000:.1f} km above the surface, "
            f"{info['mars_rel_speed']:.0f} m/s relative, "
            f"{info['mars_site_distance'] / 1000:.0f} km from the site"
        )
    elif "moon_distance" in info:
        mvx, mvy = env._moon_velocity()
        rel = math.hypot(env._vx - mvx, env._vy - mvy)
        print(
            f"moon: {(info['moon_distance'] - env.moon.radius) / 1000:.1f} km "
            f"above the surface, {rel:.0f} m/s relative, "
            f"{info['moon_site_distance'] / 1000:.0f} km from the site"
        )
    print(
        f"delta-v used {info['dv_used']:.0f} m/s of {env.total_delta_v():.0f} m/s budget "
        f"({info['dv_remaining']:.0f} m/s still in the tank)"
    )
    print(f"episode return: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
