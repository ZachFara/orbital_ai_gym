"""Pure two-body kinematics shared by the missions, the bodies and the pilots.

Everything here is a function of numbers; nothing imports an environment.
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def wrap_pi(angle: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return (angle + math.pi) % TWO_PI - math.pi


def elements(
    mu: float, body_radius: float, rx: float, ry: float, vx: float, vy: float
) -> tuple[float, float, float, float]:
    """Orbital elements of a body-relative state:
    (periapsis_alt, apoapsis_alt, specific_energy, eccentricity).

    Apoapsis is +inf on escape trajectories.
    """
    rn = math.hypot(rx, ry)
    v2 = vx * vx + vy * vy
    energy = 0.5 * v2 - mu / rn
    h = rx * vy - ry * vx
    ecc2 = 1.0 + 2.0 * energy * h * h / (mu * mu)
    ecc = math.sqrt(max(0.0, ecc2))
    if energy >= 0.0:
        peri = (h * h / mu) / (1.0 + ecc) - body_radius if ecc > 0 else rn - body_radius
        return peri, math.inf, energy, ecc
    sma = -mu / (2.0 * energy)
    return sma * (1.0 - ecc) - body_radius, sma * (1.0 + ecc) - body_radius, energy, ecc


def impact_point(
    mu: float,
    body_radius: float,
    px: float,
    py: float,
    wx: float,
    wy: float,
    site_angle: float,
) -> tuple[bool, float]:
    """Where the body-relative conic through (p, w) meets the surface.

    Returns (impacting, miss): whether the coasting trajectory ahead of the
    rocket reaches r = body_radius, and the signed angle around the body
    from `site_angle` (a world-frame angle) to that surface point, or to the
    next periapsis point when it does not reach. Two-body in the body's
    frame: exact for a terminal descent, bent by third bodies over long
    coasts. An instrument, not an oracle.
    """
    r = math.hypot(px, py)
    theta_r = math.atan2(py, px)
    if r <= body_radius:
        return True, wrap_pi(theta_r - site_angle)
    h = px * wy - py * wx
    v2 = wx * wx + wy * wy
    rv = px * wx + py * wy
    ex = ((v2 - mu / r) * px - rv * wx) / mu
    ey = ((v2 - mu / r) * py - rv * wy) / mu
    ecc = math.hypot(ex, ey)
    if ecc < 1e-9 or abs(h) < 1.0:
        # circular never lands; a radial drop lands straight below
        return abs(h) < 1.0, wrap_pi(theta_r - site_angle)
    p_param = h * h / mu
    f0 = math.copysign(math.acos(max(-1.0, min(1.0, (p_param / r - 1.0) / ecc))), rv)
    sgn = math.copysign(1.0, h)
    cos_fi = (p_param / body_radius - 1.0) / ecc
    if cos_fi > 1.0:
        # periapsis clears the surface: no impact ahead
        if 0.5 * v2 - mu / r >= 0.0 and rv > 0.0:
            return False, wrap_pi(theta_r - site_angle)  # outbound hyperbola
        return False, wrap_pi(theta_r + sgn * (-f0 % TWO_PI) - site_angle)
    # the conic dips below the surface at true anomaly -fi (rising back out
    # at +fi); the next crossing forward of the rocket's anomaly f0 is impact
    fi = math.acos(max(-1.0, min(1.0, cos_fi)))
    return True, wrap_pi(theta_r + sgn * ((-fi - f0) % TWO_PI) - site_angle)
