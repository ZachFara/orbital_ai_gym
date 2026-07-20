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
from orbital_gym.missions.to_orbit import TWO_PI, TwoStageToOrbitEnv, _wrap_pi
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_moon import ToMoonEnv


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
        if env.attitude_control:
            # pitch is elevation above the local horizon, so the desired
            # direction is sin(pitch) radial + cos(pitch) tangential
            return np.array([2.0 * throttle - 1.0, math.sin(pitch), math.cos(pitch)],
                            dtype=np.float32)
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

    PERI_FRAC = 1.03       # parking-orbit periapsis, * atmosphere cutoff
    LEVEL_FRAC = 1.04      # level-burn altitude, * cutoff (must exceed PERI_FRAC)
    TURN_END_FRAC = 0.5    # pitch schedule reaches horizontal at this * cutoff
    TURN_EXP = 0.78
    TLI_BAND = (0.995, 1.005)  # accepted transfer apoapsis, * moon orbit radius
    BRAKE_MARGIN = 0.93    # accel fraction the stopping-distance trigger assumes
    BRAKE_LEAD = 1.28      # stopping distance vs radial gap; center of the
                           # landing plateau [1.20, 1.35] on the moon rocket
                           # (scanned; the Keplerian moon's arrival is slow
                           # and shallow, so the flight path runs well past
                           # the radial gap)
    GATE_ALT = 4_000.0     # descend handoff altitude scale, m
    SITE_REACH = 30_000.0  # only steer for the site when within this, m

    def __init__(self, env: ToMoonEnv):
        self.env = env
        self.mode = "ascend"
        p = env.planet
        self.peri_target = self.PERI_FRAC * p.space_altitude
        self.h_ref = self.LEVEL_FRAC * p.space_altitude
        self.turn_end = self.TURN_END_FRAC * p.space_altitude
        self._ra_target = 0.0
        self._burning = False
        self._trim = False

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
            return np.array([2.0 * throttle - 1.0, wx * ux + wy * uy, wx * tx + wy * ty],
                            dtype=np.float32)
        turn = np.clip(err / (e.rocket.max_turn_rate * e.step_duration), -1.0, 1.0)
        return np.array([2.0 * throttle - 1.0, turn], dtype=np.float32)

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
                return self._act_elev(math.atan2(a_v, a_h),
                                      min(1.0, math.hypot(a_v, a_h) / a_av),
                                      u, tw)

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
                        math.sqrt(1.0 + e) * math.cos(nu / 2.0))
                    t_x = (ecc_an - e * math.sin(ecc_an)) / math.sqrt(
                        p.mu / a_sma ** 3)
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
                return self._act_elev(prograde, 0.3 if self._trim else 0.0,
                                      u, tw, gate=0.2)

        if self.mode == "tli":
            # burn prograde up to the transfer-ellipse speed, throttle
            # tapered (apoapsis is far too cutoff-sensitive for 1 s steps)
            v_go = math.sqrt(p.mu * (2.0 / rn - 2.0 / (rn + self._ra_target))) \
                - math.hypot(float(env._vx), float(env._vy))
            if v_go <= 0.0:
                self.mode = "transfer"
            else:
                return self._act_elev(prograde, min(1.0, max(0.05, v_go / 25.0)),
                                      u, tw, gate=0.3)

        prel = r - mpos
        vrel = v - mvel
        d = float(np.linalg.norm(prel))

        if self.mode == "transfer":
            if d < 6.0 * moon.radius:
                self.mode = "brake"
            else:
                return self._act_elev(prograde, 0.0, u, tw)

        if self.mode == "brake":
            malt = d - moon.radius
            speed = float(np.linalg.norm(vrel))
            if (speed < 70.0 and malt < 8.0 * self.GATE_ALT) \
                    or malt < 1.2 * self.GATE_ALT:
                self.mode = "descend"
            else:
                return self._brake(info, prel, vrel, d)

        return self._descend(info, prel, vrel, d)

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
        disc = pv * pv - vv * (float(prel @ prel) - moon.radius ** 2)
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
        if pv < 0.0 and v * v / (2.0 * a_net) >= \
                self.BRAKE_LEAD * max(1.0, d - moon.radius):
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
        x_err = _wrap_pi(math.atan2(site[1], site[0])
                         - math.atan2(prel[1], prel[0])) * d
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
        return self._act_world(math.atan2(a_cmd[1], a_cmd[0]),
                               min(1.0, float(np.linalg.norm(a_cmd)) / a_av))


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
    ap.add_argument("--mission", default="orbit", choices=["orbit", "return", "moon"])
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
    if "moon_distance" in info:
        mvx, mvy = env._moon_velocity()
        rel = math.hypot(env._vx - mvx, env._vy - mvy)
        print(f"moon: {(info['moon_distance'] - env.moon.radius) / 1000:.1f} km "
              f"above the surface, {rel:.0f} m/s relative, "
              f"{info['moon_site_distance'] / 1000:.0f} km from the site")
    print(f"delta-v used {info['dv_used']:.0f} m/s of {env.total_delta_v():.0f} m/s budget "
          f"({info['dv_remaining']:.0f} m/s still in the tank)")
    print(f"episode return: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
