"""Two-body Lambert problem."""

from __future__ import annotations

import math

import numpy as np

from ..missions.to_orbit import TWO_PI


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
