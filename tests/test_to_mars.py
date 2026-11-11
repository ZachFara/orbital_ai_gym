import math

import numpy as np
import pytest
import gymnasium as gym

from orbital_gym.missions.to_mars import ToMarsEnv

IDLE = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # throttle 0, hold attitude


def _empty_tank(env):
    env._stage = len(env.rocket.stages) - 1
    env._fuel = 0.0
    env._mass = env.rocket.stages[-1].dry_mass


def test_registered_id_and_obs_size():
    env = gym.make("ToMars-v0")
    obs, info = env.reset()
    assert obs.shape == (43,)
    assert info["soi"] == "planet"
    assert len(env.unwrapped.rocket.stages) == 3
    env.close()


def test_rails_are_keplerian_about_the_star():
    env = ToMarsEnv()
    mu = env.star.mu
    assert env.system.home_orbit_period == pytest.approx(
        2.0 * math.pi * math.sqrt(env.home_r ** 3 / mu), rel=1e-9)
    assert env.target.orbit_period == pytest.approx(
        2.0 * math.pi * math.sqrt(env.target.orbit_radius ** 3 / mu), rel=1e-9)


def test_tidal_acceleration_vanishes_at_the_planet():
    """The star and the target enter as the pull on the rocket minus the pull
    on the planet, so a rocket at the planet's centre feels nothing extra:
    the planet frame is a free-falling frame, not a wrong one."""
    env = ToMarsEnv()
    env.reset()
    ax, ay = env._extra_acceleration(0.0, 0.0, 0.0, 0.0)
    assert ax == 0.0 and ay == 0.0
    # and at the pad the tide is small next to gravity (percent level)
    ax, ay = env._extra_acceleration(0.0, env.planet.radius, 0.0, 0.0)
    assert math.hypot(ax, ay) < 0.03 * env.planet.surface_gravity


def test_heliocentric_coast_conserves_energy_far_from_the_planets():
    env = ToMarsEnv()
    env.reset()
    R = 1.25 * env.target.orbit_radius
    ang = env._home_angle() + math.pi          # opposite side of the star
    hx, hy = env._home_helio()
    hvx, hvy = env._home_helio_velocity()
    s = math.sqrt(env.star.mu / R)
    env._teleport(R * math.cos(ang) - hx, R * math.sin(ang) - hy,
                  -s * math.sin(ang) - hvx, s * math.cos(ang) - hvy)
    _empty_tank(env)
    e0 = env._helio_elements()[2]
    for _ in range(3000):
        _, _, term, trunc, info = env.step(IDLE)
        assert not term
    assert info["soi"] == "sun"
    e1 = env._helio_elements()[2]
    assert abs(e1 - e0) / abs(e0) < 0.01


def test_target_frame_switch_and_thin_atmosphere():
    """Inside the target's SOI the published elements are target-relative,
    the impact predictor sees the surface, and a free fall from 5 km is
    braked a little by the thin atmosphere."""
    env = ToMarsEnv()
    env.reset()
    mx, my = env._mars_world()
    mn = math.hypot(mx, my)
    d = env.target.radius + 5_000.0
    env._teleport(mx + mx / mn * d, my + my / mn * d, *env._mars_velocity())
    _empty_tank(env)
    _, _, terminated, _, info = env.step(IDLE)
    assert info["soi"] == "mars"
    assert 0.0 < info["apoapsis"] < 20_000.0
    assert info["mars_impact"] is True
    n = 1
    while not terminated:
        _, _, terminated, truncated, info = env.step(IDLE)
        assert not truncated
        n += 1
    assert info["termination"] == "crashed"
    g = env.target.mu / env.target.radius ** 2
    airless = math.sqrt(2.0 * g * 5_000.0)
    assert 0.85 * airless < info["mars_rel_speed"] < airless


def test_escape_guard_is_heliocentric():
    """Being hyperbolic relative to the planet is the point of the transfer;
    only a star-hyperbolic coast with an empty tank counts as escaped."""
    env = ToMarsEnv()
    env.reset()
    # planet-hyperbolic but star-bound: 2 planet-SOI out, moving with the planet
    env._teleport(2.0 * env.planet_soi, 0.0, 300.0, 0.0)
    _empty_tank(env)
    _, _, terminated, _, info = env.step(IDLE)
    assert not terminated
    assert info["soi"] == "sun"
    # star-hyperbolic: give it twice the star's escape speed
    x, y, *_ = env.rocket_helio()
    v_esc = math.sqrt(2.0 * env.star.mu / math.hypot(x, y))
    hvx, hvy = env._home_helio_velocity()
    env._teleport(env._rx, env._ry, 2.0 * v_esc - hvx, -hvy)
    _, _, terminated, _, info = env.step(IDLE)
    assert terminated and info["termination"] == "escaped"


def test_mars_autopilot_lands_softly_on_the_target():
    """The scripted pilot flies the whole profile: ascent, hold, simulate-
    ahead departure through the star's tide, Lambert midcourse corrections,
    capture into a parking orbit, deorbit and powered descent. It lands soft,
    well off the site -- putting it on the site is the learning problem."""
    from scripts.autopilot_demo import MarsAutopilot

    env = ToMarsEnv()
    obs, info = env.reset()
    pilot = MarsAutopilot(env)
    while True:
        obs, _, terminated, truncated, info = env.step(pilot(info))
        if terminated or truncated:
            break
    assert info["transfer"] is True
    assert info["termination"] in ("landed_mars", "landed_mars_site")
    assert info["mars_rel_speed"] <= env.mars_landing_speed
    assert info["dv_remaining"] > 0.0
