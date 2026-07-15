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
