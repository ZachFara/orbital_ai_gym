import numpy as np
import pytest
import gymnasium as gym

from orbital_gym.missions.to_moon import ToMoonEnv

IDLE = np.array([-1.0, 0.0], dtype=np.float32)  # throttle 0, no turn
SPIN = np.array([-1.0, 1.0], dtype=np.float32)  # throttle 0, full turn


def test_registered_id_and_obs_size():
    env = gym.make("ToMoon-v0")
    obs, _ = env.reset()
    assert obs.shape == (15,)
    env.close()


def test_turn_penalty_is_inherited():
    env = ToMoonEnv()
    env.reset()
    total = 0.0
    for _ in range(20):
        _, r, *_ = env.step(SPIN)
        total += r
    assert total == pytest.approx(-20.0 * env.turn_penalty * env.step_duration, rel=1e-6)


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


def test_moon_autopilot_reaches_and_brakes_at_the_moon():
    """The scripted pilot flies the full profile: hold, transfer, braking.

    The rocket's delta-v budget is razor thin for this mission; the pilot
    brakes to a slow near-touchdown a few km up but runs the tank dry, so a
    soft landing is not asserted here (train_guided.py exists to close that
    gap). What it must do: complete the orbit hold, cross to the moon, and
    kill most of the ~2 km/s arrival speed close to the surface.
    """
    import math

    from scripts.autopilot_demo import MoonAutopilot

    env = ToMoonEnv()
    obs, info = env.reset()
    pilot = MoonAutopilot(env)
    min_moon_d = math.inf
    while True:
        obs, _, terminated, truncated, info = env.step(pilot(info))
        min_moon_d = min(min_moon_d, info["moon_distance"])
        if terminated or truncated:
            break
    assert info["transfer"] is True  # orbit hold completed, phase 2 entered
    assert min_moon_d < env.moon.radius + 80_000.0  # reached the moon
    mvx, mvy = env._moon_velocity()
    rel_speed = math.hypot(env._vx - mvx, env._vy - mvy)
    assert rel_speed < 500.0  # braked from ~2.1 km/s arrival
    assert info["moon_distance"] < env.moon.radius + 20_000.0  # ...down low
