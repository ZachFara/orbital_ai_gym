"""CelestialBody and CircularRails: the one abstraction for bodies on rails."""

import math

import pytest

from orbital_gym.bodies import Atmosphere, CelestialBody, CircularRails
from orbital_gym.kinematics import impact_point, wrap_pi
from orbital_gym.presets import MOONS, PLANETS, SYSTEMS


def test_rails_velocity_is_the_derivative_of_position() -> None:
    rails = CircularRails(radius=2.4e6, period=12_430.0, phase0=0.3)
    t, h = 1000.0, 1e-3
    x0, y0 = rails.position(t - h)
    x1, y1 = rails.position(t + h)
    vx, vy = rails.velocity(t)
    assert vx == pytest.approx((x1 - x0) / (2 * h), rel=1e-6)
    assert vy == pytest.approx((y1 - y0) / (2 * h), rel=1e-6)
    assert math.hypot(vx, vy) == pytest.approx(rails.rate * rails.radius)


def test_body_relative_to_a_moving_frame() -> None:
    """A body seen from a frame that itself moves: position and velocity are
    differences of the two rails, and a star at the primary is minus the
    frame's own rails."""
    system = SYSTEMS["easy"]
    home = CircularRails(system.home_orbit_radius, system.home_orbit_period)
    mars = CelestialBody.from_preset("mars", "planet", system.target, system.star.mu, frame=home)
    sun = CelestialBody("sun", "star", system.star.radius, system.star.mu, math.inf, frame=home)
    t = 2500.0
    assert mars.rails is not None
    mx, my = mars.rails.position(t)
    hx, hy = home.position(t)
    assert mars.position(t) == pytest.approx((mx - hx, my - hy))
    assert sun.position(t) == pytest.approx((-hx, -hy))
    hvx, hvy = home.velocity(t)
    assert sun.velocity(t) == pytest.approx((-hvx, -hvy))


def test_tidal_gravity_vanishes_at_the_frame_origin_and_plain_gravity_does_not() -> None:
    system = SYSTEMS["easy"]
    home = CircularRails(system.home_orbit_radius, system.home_orbit_period)
    tidal = CelestialBody("sun", "star", 1.0, system.star.mu, math.inf, frame=home, tidal=True)
    plain = CelestialBody("sun", "star", 1.0, system.star.mu, math.inf, frame=home, tidal=False)
    assert tidal.gravity(0.0, 0.0, 0.0) == (0.0, 0.0)
    gx, gy = plain.gravity(0.0, 0.0, 0.0)
    assert math.hypot(gx, gy) == pytest.approx(system.star.mu / system.home_orbit_radius**2)


def test_sphere_of_influence_is_the_laplace_radius() -> None:
    planet = PLANETS["easy"]
    moon = CelestialBody.from_preset("moon", "moon", MOONS["easy"], planet.mu)
    expected = MOONS["easy"].orbit_radius * (MOONS["easy"].mu / planet.mu) ** 0.4
    assert moon.soi == pytest.approx(expected)
    assert moon.atmosphere is None
    assert moon.density(1_000.0) == 0.0


def test_atmosphere_and_drag_act_on_the_body_relative_velocity() -> None:
    atmo = Atmosphere(rho0=1.0, scale_height=1_000.0, space_altitude=10_000.0)
    rails = CircularRails(radius=1.0e7, period=20_000.0)
    body = CelestialBody("p", "planet", 1.0e5, 1.0e11, 1.0e6, rails=rails, atmosphere=atmo)
    t = 0.0
    bx, by = body.position(t)
    bvx, bvy = body.velocity(t)
    # co-moving with the body, 500 m up: no relative speed, no drag
    assert body.drag(bx + body.radius + 500.0, by, bvx, bvy, t, 10.0, 1000.0) == (0.0, 0.0)
    # moving 100 m/s radially outward relative to it: drag points back inward
    ax, ay = body.drag(bx + body.radius + 500.0, by, bvx + 100.0, bvy, t, 10.0, 1000.0)
    assert ax < 0.0 and ay == pytest.approx(0.0)
    assert atmo.density(20_000.0) == 0.0
    assert atmo.density(0.0) == pytest.approx(1.0)


def test_impact_point_of_a_vertical_drop_is_straight_below() -> None:
    mu, radius = 2.4e11, 160_000.0
    # 5 km above the surface at angle 0.4, no horizontal speed
    r = radius + 5_000.0
    px, py = r * math.cos(0.4), r * math.sin(0.4)
    impacting, miss = impact_point(mu, radius, px, py, 0.0, 0.0, site_angle=0.1)
    assert impacting
    assert miss == pytest.approx(wrap_pi(0.4 - 0.1))
    # a fast horizontal pass well above the surface never reaches it
    v_circ = math.sqrt(mu / r)
    impacting, _ = impact_point(
        mu, radius, px, py, -v_circ * math.sin(0.4), v_circ * math.cos(0.4), 0.0
    )
    assert not impacting
