import gymnasium as gym
import numpy as np
import pytest

from orbital_gym.missions.to_moon import ToMoonEnv

IDLE = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # throttle 0, hold attitude
SPIN = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # throttle 0, point at horizon


def test_registered_id_and_obs_size():
    env = gym.make("ToMoon-v0")
    obs, _ = env.reset()
    assert obs.shape == (30,)
    env.close()


def test_moon_rails_are_keplerian():
    """The on-rails period must match Kepler for the orbit radius. Anything
    else makes the moon-relative frame non-inertial: a lander fights a
    phantom v^2/r - mu/r^2 force (the old arbitrary 7200 s period put
    1.2 m/s^2 of it under every landing, and made anything co-moving with
    the moon planet-hyperbolic).
    """
    import math

    env = ToMoonEnv()
    kepler = 2.0 * math.pi * math.sqrt(env.moon.orbit_radius**3 / env.planet.mu)
    assert env.moon.orbit_period == pytest.approx(kepler, rel=1e-9)


def test_turn_penalty_is_inherited():
    env = ToMoonEnv()
    env.reset()
    total = 0.0
    for _ in range(5):  # 5 steps of full-rate slew toward the horizon
        _, r, *_ = env.step(SPIN)
        total += r
    assert total == pytest.approx(-5.0 * env.turn_penalty * env.step_duration, rel=1e-6)


def test_orbit_progress_requires_stable_orbit():
    env = ToMoonEnv()
    env.reset()
    p = env.planet
    alt = 200_000.0
    env._teleport(0.0, p.radius + alt, 0.4 * p.circular_velocity(alt), 100.0)
    for _ in range(20):
        _, _, _, _, info = env.step(IDLE)
    assert info["orbit"] is False
    assert info["orbit_progress"] == pytest.approx(0.0, abs=1e-12)


def test_moon_extra_acceleration_is_nonzero():
    env = ToMoonEnv()
    env.reset()
    ax, ay = env._extra_acceleration(env._rx, env._ry, env._vx, env._vy)
    assert abs(ax) + abs(ay) > 0.0


def test_soi_frame_switch_and_impact_predictor():
    """Inside the SOI, info's elements must go moon-relative, the escape
    guard must stay quiet (on a fast-railed moon a braking lander is
    outright planet-hyperbolic; the guard used to fire "escaped" moments
    before touchdown), the impact predictor must see the surface coming,
    and the episode must run to the actual impact.
    """
    import math

    env = ToMoonEnv()
    env.reset()
    # co-moving with the moon, 5 km above its surface (directly above the
    # site: the test drop point is on the far face, where the site sits),
    # tank dry
    mx, my = env._moon_world()
    mn = math.hypot(mx, my)
    d = env.moon.radius + 5_000.0
    env._teleport(mx + mx / mn * d, my + my / mn * d, *env._moon_velocity())
    env._stage = len(env.rocket.stages) - 1
    env._fuel = 0.0

    _obs, _, terminated, _, info = env.step(IDLE)
    assert info["soi"] == "moon"
    assert not terminated
    # moon-relative elements: dropped from rest, the apoapsis is the release
    # altitude, not some planet-scale number
    assert 0.0 < info["apoapsis"] < 20_000.0
    # a straight-down drop over the site: impact predicted, right at the site
    assert info["moon_impact"] is True
    assert abs(info["moon_impact_miss"]) < 5_000.0

    while not terminated:
        _, _, terminated, truncated, info = env.step(IDLE)
        assert not truncated
    assert info["termination"] == "crashed"  # free fall from 5 km is ~300 m/s


def test_moon_autopilot_soft_lands_on_the_moon():
    """The scripted pilot flies the full profile on the "moon" rocket:
    hold, transfer, retro braking, and a soft touchdown. It lands well off
    the site -- putting it ON the site is the learning problem
    (train_guided.py), not the autopilot's job.
    """
    from scripts.autopilot_demo import MoonAutopilot

    env = ToMoonEnv()
    _obs, info = env.reset()
    pilot = MoonAutopilot(env)
    while True:
        _obs, _, terminated, truncated, info = env.step(pilot(info))
        if terminated or truncated:
            break
    assert info["transfer"] is True  # orbit hold completed, phase 2 entered
    assert info["termination"] in ("landed_moon", "landed_moon_site")
    assert info["moon_rel_speed"] <= env.moon_landing_speed
    assert info["dv_remaining"] > 0.0  # landed under power, not on fumes
