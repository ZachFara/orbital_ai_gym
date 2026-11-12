"""Smoke test: every registered env constructs, resets, steps and reports
`info["success"]`; the practice id's knobs reach the wrapper."""

from typing import Any

import gymnasium as gym
import numpy as np

import orbital_gym

ENV_IDS = (
    "TwoStageToOrbit-v0",
    "OrbitAndReturn-v0",
    "ToMoon-v0",
    "ToMars-v0",
    "MoonLandingFromOrbit-v0",
)


def test_package_imports_and_registers_every_mission() -> None:
    assert orbital_gym.__version__
    for env_id in ENV_IDS:
        env = gym.make(env_id)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert info["termination"] is None
        assert info["success"] is False
        assert env.action_space.shape is not None
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not terminated and not truncated
        assert info["success"] is False
        env.close()


def test_practice_id_knobs_and_a_short_episode() -> None:
    full = gym.make("MoonLandingFromOrbit-v0")
    easy = gym.make("MoonLandingFromOrbit-v0", level=0.0, extra_obs=False, site_noise=0.0)
    full_u: Any = full.unwrapped
    easy_u: Any = easy.unwrapped
    assert full_u.observation_space.shape[0] - easy_u.observation_space.shape[0] == 8  # guidance
    assert easy_u.level == 0.0 and full_u.level == 1.0
    assert easy_u.throttle_map == "exp" and easy_u.rc.w_miss == 2.0
    # level 0 starts in the last metres before touchdown: engine off ends it fast
    easy.reset(seed=1)
    off = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(200):
        _, _, terminated, truncated, info = easy.step(off)
        if terminated or truncated:
            break
    assert terminated or truncated
    assert info["termination"] in ("crashed", "landed_moon", "landed_moon_site")
    assert info["success"] == (info["termination"] == "landed_moon_site")
    full.close()
    easy.close()
