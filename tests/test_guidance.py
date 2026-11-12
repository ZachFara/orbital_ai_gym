"""Guidance: the conic impact predictor against brute-force coasting, and the
zero-effort miss gauge's behaviour across the orbit / burn / fall regimes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from orbital_gym.guidance import Guidance, conic_impact
from orbital_gym.missions.to_moon import ToMoonEnv

OFF = np.array([-1.0, 0.0, 0.0], dtype=np.float32)


def _place(env: Any, altitude: float, vr: float, vt: float) -> None:
    """Put the rocket `altitude` above the moon's near face with the given
    moon-relative radial and tangential speeds, tank empty."""
    body = env.target_body
    mx, my = body.position(env._time)
    mvx, mvy = body.velocity(env._time)
    d = body.radius + altitude
    ux, uy = mx / math.hypot(mx, my), my / math.hypot(mx, my)
    env._teleport(mx + ux * d, my + uy * d, mvx + ux * vr + uy * vt, mvy + uy * vr - ux * vt)
    env._stage = len(env.rocket.stages) - 1
    env._fuel = 0.0
    env._mass = env.rocket.stages[-1].dry_mass


@pytest.mark.parametrize(
    ("altitude", "vr", "vt"), [(5_000.0, -50.0, 0.0), (40_000.0, -200.0, 300.0)]
)
def test_conic_impact_matches_brute_force_coasting(altitude: float, vr: float, vt: float) -> None:
    env: Any = ToMoonEnv()
    env.reset()
    _place(env, altitude, vr, vt)
    g = Guidance.from_env(env, site_angle=0.0)
    assert g.impact.reaches
    n = 0
    while True:
        _, _, terminated, truncated, info = env.step(OFF)
        n += 1
        assert not truncated
        if terminated:
            break
    assert info["termination"] == "crashed"
    assert n == pytest.approx(g.impact.dt, abs=2.0)
    assert info["moon_rel_speed"] == pytest.approx(g.impact.speed, rel=0.02)


def test_orbit_reads_the_virtual_miss_and_a_fall_reads_the_actual_one() -> None:
    env: Any = ToMoonEnv()
    env.reset()
    body = env.target_body
    # circular orbit 60 km up: the path never reaches the surface, so the
    # gauge is the stop-and-drop virtual miss and the flag is False
    v_circ = math.sqrt(body.mu / (body.radius + 60_000.0))
    _place(env, 60_000.0, 0.0, v_circ)
    g = Guidance.from_env(env, site_angle=0.0)
    assert not g.impact.reaches
    miss, reaches = g.zem
    assert not reaches
    assert miss == pytest.approx(g.virtual_miss)
    assert g.channels()[7] == 0.0
    # a vertical fall from 20 km: reaches, and the blended miss IS the actual miss
    _place(env, 20_000.0, -100.0, 0.0)
    g = Guidance.from_env(env, site_angle=0.0)
    miss, reaches = g.zem
    assert reaches
    assert miss == pytest.approx(g.actual_miss)
    assert g.stop_ratio > 0.0
    assert g.brake_dv > g.impact.speed  # gravity loss included
    assert g.channels().shape == (Guidance.N_CHANNELS,)


def test_instruments_are_zero_outside_the_sphere_of_influence() -> None:
    env: Any = ToMoonEnv()
    env.reset()  # on the launch pad, far from the moon
    g = Guidance.from_env(env, site_angle=0.0)
    assert not g.in_soi
    assert np.all(g.channels() == 0.0)
    assert g.miss_potential(2.0, 20_000.0) == 0.0


def test_radial_drop_time_is_the_free_fall_time() -> None:
    mu, radius, h = 2.4e11, 160_000.0, 5_000.0
    imp = conic_impact(mu, radius, radius + h, 0.0, 0.0, 0.0)
    assert imp.reaches
    g = mu / (radius + h) ** 2
    assert imp.dt == pytest.approx(math.sqrt(2.0 * h / g), rel=0.1)
    assert imp.vr < 0.0 and imp.vt == 0.0
