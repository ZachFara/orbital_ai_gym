"""Smoke test: every registered mission constructs, resets and steps."""

import gymnasium as gym
import numpy as np

import orbital_gym

ENV_IDS = ("TwoStageToOrbit-v0", "OrbitAndReturn-v0", "ToMoon-v0", "ToMars-v0")


def test_package_imports_and_registers_every_mission() -> None:
    assert orbital_gym.__version__
    for env_id in ENV_IDS:
        env = gym.make(env_id)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert info["termination"] is None
        assert env.action_space.shape is not None
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not terminated and not truncated
        env.close()
