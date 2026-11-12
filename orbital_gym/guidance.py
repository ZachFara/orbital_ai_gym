"""Zero-effort guidance instruments for a powered landing on a body on rails.

`Guidance` is built from a body and the rocket's state and answers the
questions a lander's guidance computer answers: where and how fast the
coasting trajectory hits the surface (`ConicImpact`), the signed miss
between the predicted touchdown and the site -- blended with the touchdown
of a standard stop-and-drop while the path does not yet reach the surface,
so the gauge is continuous through the retro burn -- and whether full thrust
can still stop the fall. The RL task uses one `Guidance` for both its
observation channels and its shaping potential, so the two can never drift
apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .bodies import CelestialBody
from .kinematics import TWO_PI, wrap_pi


@dataclass(frozen=True)
class ConicImpact:
    """The coasting conic ahead of the rocket, met with the surface."""

    reaches: bool  # does the path ahead reach r = body radius?
    dt: float  # seconds until it does (or until the next periapsis; inf if never)
    speed: float  # speed there (vis-viva)
    vr: float  # radial component there (negative = downward)
    vt: float  # tangential component there (angular momentum)


def conic_impact(mu: float, R: float, px: float, py: float, wx: float, wy: float) -> ConicImpact:
    """Coast the body-relative state (p, w) along its two-body conic to the
    surface. These are the zero-effort-miss / zero-effort-velocity numbers
    that lunar-descent guidance laws act on."""
    r = math.hypot(px, py)
    v2 = wx * wx + wy * wy
    rv = px * wx + py * wy
    h = px * wy - py * wx
    ex = ((v2 - mu / r) * px - rv * wx) / mu
    ey = ((v2 - mu / r) * py - rv * wy) / mu
    e = math.hypot(ex, ey)
    pp = h * h / mu
    if r <= R:
        return ConicImpact(True, 0.0, math.sqrt(v2), rv / max(r, 1.0), h / max(r, 1.0))
    if e < 1e-9 or pp < 1e-6:
        if pp < 1e-6:  # radial drop: straight down, dt from energy
            vimp = math.sqrt(max(0.0, v2 + 2.0 * mu * (1.0 / R - 1.0 / r)))
            dt = 2.0 * (r - R) / max(1e-3, vimp + max(0.0, -rv / r)) if rv <= 0 else math.inf
            return ConicImpact(rv <= 0.0, dt, vimp, -vimp, 0.0)
        return ConicImpact(False, math.inf, math.sqrt(v2), 0.0, math.sqrt(v2))
    f0 = math.copysign(math.acos(max(-1.0, min(1.0, (pp / r - 1.0) / e))), rv)
    cos_fi = (pp / R - 1.0) / e
    reaches = cos_fi <= 1.0
    if reaches:
        fi = math.acos(max(-1.0, min(1.0, cos_fi)))
        ft = -fi  # descending crossing of r = R
        if e >= 1.0 and f0 > ft:  # hyperbolic and already past it: outbound
            return ConicImpact(False, math.inf, math.sqrt(v2), 0.0, math.sqrt(v2))
        r_t = R
    else:
        ft = 0.0  # next periapsis
        if e >= 1.0 and rv > 0.0:
            return ConicImpact(False, math.inf, math.sqrt(v2), 0.0, math.sqrt(v2))
        r_t = pp / (1.0 + e)
    if e < 1.0:
        a = pp / (1.0 - e * e)
        n = math.sqrt(mu / (a * a * a))

        def mean_anom(f: float) -> float:
            ecc_anom = 2.0 * math.atan2(
                math.sqrt(1.0 - e) * math.sin(f / 2.0), math.sqrt(1.0 + e) * math.cos(f / 2.0)
            )
            return ecc_anom - e * math.sin(ecc_anom)

        dt = ((mean_anom(ft) - mean_anom(f0)) % TWO_PI) / n
    else:
        a = pp / (e * e - 1.0)
        n = math.sqrt(mu / (a * a * a))

        def mean_anom(f: float) -> float:
            t = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(f / 2.0)
            t = max(-0.999999, min(0.999999, t))
            hyp_anom = 2.0 * math.atanh(t)
            return e * math.sinh(hyp_anom) - hyp_anom

        dt = max(0.0, (mean_anom(ft) - mean_anom(f0)) / n)
    vimp = math.sqrt(max(0.0, v2 + 2.0 * mu * (1.0 / r_t - 1.0 / r)))
    vt = h / r_t
    vr = -math.sqrt(max(0.0, vimp * vimp - vt * vt))
    return ConicImpact(reaches, dt, vimp, vr, vt)


class Guidance:
    """Landing instruments for one rocket state relative to one body.

    `a_max` is the thrust acceleration available now, `dv_remaining` the
    ideal delta-v left. Everything is computed once per instance.
    """

    N_CHANNELS = 8
    BLEND_SPEED = 300.0  # m/s of horizontal speed over which virtual -> actual miss
    BURN_FRACTION = 0.9  # of a_max assumed for the stop-and-drop burn
    FALL_FACTOR = 1.2  # powered descent takes longer than free fall

    def __init__(
        self,
        body: CelestialBody,
        t: float,
        rx: float,
        ry: float,
        vx: float,
        vy: float,
        site_angle: float,
        a_max: float,
        dv_remaining: float,
    ):
        self.body = body
        self.t = t
        self.site_angle = site_angle
        self.a_max = a_max
        self.dv_remaining = dv_remaining
        self.px, self.py, self.wx, self.wy = body.relative(rx, ry, vx, vy, t)
        self.d = max(1.0, math.hypot(self.px, self.py))
        self.altitude = self.d - body.radius
        self.in_soi = self.d <= body.soi
        self.g = body.mu / (self.d * self.d)
        self.rotation_rate = body.rails.rate if body.rails is not None else 0.0

    @classmethod
    def from_env(cls, env: Any, site_angle: float) -> Guidance:
        """From a mission env with a `target_body` (ToMoonEnv, ToMarsEnv)."""
        return cls(
            env.target_body,
            env._time,
            env._rx,
            env._ry,
            env._vx,
            env._vy,
            site_angle,
            env.rocket.stages[env._stage].thrust / env._mass,
            env.delta_v_remaining(),
        )

    # -- the coasting conic ----------------------------------------------------------

    @cached_property
    def impact(self) -> ConicImpact:
        return conic_impact(self.body.mu, self.body.radius, self.px, self.py, self.wx, self.wy)

    @cached_property
    def actual_miss(self) -> float:
        """Signed surface distance (m) from the site to the conic's impact
        point, against where the site will be at impact: the body is tidally
        locked, so the site moves along the surface while the rocket falls."""
        _, miss = self.body.impact(
            self.px + self.body.position(self.t)[0],
            self.py + self.body.position(self.t)[1],
            self.wx + self.body.velocity(self.t)[0],
            self.wy + self.body.velocity(self.t)[1],
            self.t,
            self.site_angle,
        )
        return wrap_pi(miss - self.rotation_rate * self.impact.dt) * self.body.radius

    # -- the stop-and-drop profile -----------------------------------------------------

    @cached_property
    def horizontal_speed(self) -> float:
        """Signed (counter-clockwise positive) tangential speed about the body."""
        return (self.px * self.wy - self.py * self.wx) / self.d

    @cached_property
    def virtual_miss(self) -> float:
        """Signed surface distance from the site to the touchdown of a standard
        stop-and-drop started now: burn the horizontal speed to zero at
        BURN_FRACTION of a_max (drifting half the burn), fall, and the site
        moves meanwhile."""
        v_t = self.horizontal_speed
        a_h = self.BURN_FRACTION * self.a_max
        t_b = abs(v_t) / max(1.0, a_h)
        t_f = self.FALL_FACTOR * math.sqrt(2.0 * max(0.0, self.altitude) / self.g)
        theta_td = math.atan2(self.py, self.px) + (v_t / self.d) * t_b / 2.0
        theta_site_td = self.body.angle(self.t) + self.site_angle + self.rotation_rate * (t_b + t_f)
        return wrap_pi(theta_td - theta_site_td) * self.body.radius

    @cached_property
    def zem(self) -> tuple[float, bool]:
        """(signed predicted miss in m, whether the coasting path reaches the
        surface). While the path misses the body, the stop-and-drop virtual
        miss; once it reaches, a blend by horizontal speed toward the conic's
        actual impact miss, so the gauge is continuous through the retro burn
        instead of jumping from "periapsis of a circular orbit" to "grazing
        ballistic 450 km downrange" when the burn starts."""
        imp = self.impact
        if not imp.reaches or not math.isfinite(imp.dt):
            return self.virtual_miss, False
        lam = min(1.0, abs(self.horizontal_speed) / self.BLEND_SPEED)
        return lam * self.virtual_miss + (1.0 - lam) * self.actual_miss, True

    # -- can full thrust still stop it? ----------------------------------------------

    @cached_property
    def stop_ratio(self) -> float:
        """Stopping distance at full thrust over the altitude: > 1 means the
        fall can no longer be stopped before the surface."""
        a_net = max(0.5, self.a_max - self.g)
        return self.impact.speed**2 / (2.0 * a_net * max(1.0, self.altitude))

    @cached_property
    def brake_dv(self) -> float:
        """Delta-v of the retro burn that kills the impact speed, gravity loss included."""
        a_net = max(0.5, self.a_max - self.g)
        return self.impact.speed * (1.0 + self.g / a_net)

    # -- as an observation / a potential ---------------------------------------------

    def channels(self) -> np.ndarray:
        """The eight guidance channels of the RL observation; zero outside the
        body's sphere of influence, where the instruments mean nothing."""
        if not self.in_soi:
            return np.zeros(self.N_CHANNELS, np.float32)
        imp = self.impact
        m_signed, _ = self.zem
        return np.array(
            [
                imp.speed / 1_000.0,
                imp.vr / 1_000.0,
                imp.vt / 1_000.0,
                math.log1p(min(imp.dt, 20_000.0)) / 10.0,
                math.asinh(m_signed / 5_000.0) / 3.0,  # 8 km -> 0.42, 300 km -> 1.6
                min(2.0, self.stop_ratio) / 2.0,
                min(2.0, self.brake_dv / max(1.0, self.dv_remaining)) / 2.0,
                1.0 if imp.reaches else 0.0,
            ],
            np.float32,
        )

    def miss_potential(self, weight: float, scale: float) -> float:
        """The shaping potential's miss term: asinh-shaped so the orbit-to-
        touchdown swing does not dwarf the last few kilometres; zero outside
        the sphere of influence."""
        if weight <= 0.0 or not self.in_soi:
            return 0.0
        m_signed, _ = self.zem
        return -weight * math.asinh(abs(m_signed) / (0.25 * scale)) / 3.0
