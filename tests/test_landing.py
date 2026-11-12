"""The landing task spec: the moon task is the published configuration,
the Mars task is the same recipe scaled to its body, and the old module
path still resolves the bank classes (cached pickles)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from orbital_gym.rl import landing, moon_landing
from orbital_gym.rl.landing import (
    MARS_TASK,
    MOON_TASK,
    REFERENCE_REWARD,
    MarsLandingFromOrbit,
    MoonLandingFromOrbit,
    RewardConfig,
)


def test_moon_task_is_the_reference_and_mars_is_scaled() -> None:
    assert MOON_TASK.default_reward == RewardConfig()
    assert MOON_TASK.reference_reward == REFERENCE_REWARD
    mars = MARS_TASK.reference_reward
    assert (mars.w_miss, mars.fuel, mars.hover_cost) == (2.0, 0.0005, 0.02)  # the recipe
    assert mars.d_scale > REFERENCE_REWARD.d_scale and mars.v_scale < REFERENCE_REWARD.v_scale
    assert MARS_TASK.soft == ("landed_mars", "landed_mars_site")
    assert MoonLandingFromOrbit.TASK is MOON_TASK and MarsLandingFromOrbit.TASK is MARS_TASK


def test_old_module_path_resolves_the_bank_classes() -> None:
    assert moon_landing.StateBank is landing.StateBank
    assert moon_landing.BankEntry is landing.BankEntry
    assert moon_landing.MoonLandingEnv is landing.MoonLandingEnv


@pytest.mark.parametrize("env_id", ["MoonLandingFromOrbit-v0", "MarsLandingFromOrbit-v0"])
def test_nominal_episodes_start_at_parking_orbit_entry(env_id: str) -> None:
    env = gym.make(env_id, nominal=True)
    u: Any = env.unwrapped
    _, info = env.reset(seed=3, options={"site_angle": 1.0})
    assert info["start_ttl"] == u.bank.max_ttl and info["success"] is False
    assert info["periapsis"] > 0.0 and info["orbit"]  # bound and clear of the surface
    # body-relative numbers are the ones the wrapper reports
    assert info["rel_speed"] == pytest.approx(
        u.body.relative_speed(u.env._rx, u.env._ry, u.env._vx, u.env._vy, u.env._time)
    )
    assert info["site_distance"] > 0.0
    _, reward, terminated, truncated, _ = env.step(np.zeros(3, np.float32))
    assert isinstance(reward, float) and not terminated and not truncated
    env.close()
