import math

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import orbital_gym
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv

IDLE = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # throttle 0, hold attitude
SPIN = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # throttle 0, point at horizon


def make_returning(env: OrbitAndReturnEnv, rx, ry, vx, vy):
    """Put the env straight into the return phase at the given state."""
    env.reset()
    env._teleport(rx, ry, vx, vy)
    env._orbit_angle = env._required_angle
    env._returning = True
    env._last_ang = math.atan2(env._ry, env._rx)
    env._last_pad_d = env._pad_distance()


def test_gymnasium_api_conformance():
    check_env(OrbitAndReturnEnv(), skip_render_check=True)


def test_registered_id_and_obs_shape():
    env = gym.make("OrbitAndReturn-v0", planet="easy")
    obs, info = env.reset()
    assert obs.shape == (12,)
    # at reset the rocket sits on the pad: bearing 0, no hold progress
    assert obs[9] == pytest.approx(0.0, abs=1e-6)   # sin(bearing)
    assert obs[10] == pytest.approx(1.0, abs=1e-6)  # cos(bearing)
    assert obs[11] == 0.0                            # orbit progress
    assert info["returning"] is False
    env.close()


def test_return_mission_inherits_turn_penalty():
    env = OrbitAndReturnEnv()
    env.reset()
    total = 0.0
    for _ in range(5):  # 5 steps of full-rate slew toward the horizon
        _, r, *_ = env.step(SPIN)
        total += r
    assert total == pytest.approx(-5.0 * env.turn_penalty * env.step_duration, rel=1e-6)


def test_orbit_hold_progress_is_circularity_weighted():
    env = OrbitAndReturnEnv()
    env.reset()
    p = env.planet
    alt = 300_000.0
    r = p.radius + alt
    v_circ = p.circular_velocity(alt)

    # Circular orbit at the same radius: progress should accrue fastest.
    env._teleport(0.0, r, v_circ, 0.0)
    for _ in range(80):
        _, _, _, _, info_c = env.step(IDLE)
    assert info_c["orbit"] is True
    progress_c = info_c["orbit_progress"]
    circ_c = info_c["circularity"]

    # Elliptic but still stable orbit at the same point (apoapsis): 0.97 v_circ
    # puts the periapsis near 200 km, well clear of the 70 km cutoff, while
    # 0.9 v_circ would bury it in the atmosphere and stop progress entirely.
    env.reset()
    env._teleport(0.0, r, 0.97 * v_circ, 0.0)
    for _ in range(80):
        _, _, _, _, info_e = env.step(IDLE)
    assert info_e["orbit"] is True
    progress_e = info_e["orbit_progress"]
    circ_e = info_e["circularity"]

    assert circ_e < circ_c
    assert progress_e < progress_c


def test_orbit_hold_progress_requires_stable_orbit():
    env = OrbitAndReturnEnv()
    env.reset()
    p = env.planet
    alt = 200_000.0

    # In space but not a stable orbit (periapsis below ground): no hold progress.
    env._teleport(0.0, p.radius + alt, 0.4 * p.circular_velocity(alt), 100.0)
    for _ in range(30):
        _, _, _, _, info = env.step(IDLE)
    assert info["orbit"] is False
    assert info["orbit_progress"] == pytest.approx(0.0, abs=1e-12)


def test_hold_reward_sums_to_orbit_reward_then_phase_switches():
    env = OrbitAndReturnEnv()
    env.reset()
    p = env.planet
    alt = 150_000.0
    env._teleport(0.0, p.radius + alt, p.circular_velocity(alt), 0.0)
    total, info = 0.0, None
    for _ in range(2 * env.max_episode_steps):
        _, r, terminated, truncated, info = env.step(IDLE)
        total += r
        assert not terminated and not truncated
        if info["returning"]:
            break
    assert info["returning"] and info["orbit_progress"] == 1.0
    # fixed sum: exactly orbit_reward for completing the hold, however long
    # it took; nothing extra for the time spent up there
    assert total == pytest.approx(env.orbit_reward, rel=1e-3)
    # a circular hold takes about half the orbital period
    half_period = math.pi * math.sqrt((p.radius + alt) ** 3 / p.mu)
    assert info["time"] == pytest.approx(half_period, rel=0.05)


def test_descent_shaping_telescopes_and_cannot_be_farmed():
    env = OrbitAndReturnEnv()
    p = env.planet
    alt = 200_000.0
    make_returning(env, 0.0, p.radius + alt, p.circular_velocity(alt), 0.0)
    d0 = env._last_pad_d
    total = 0.0
    for _ in range(500):
        _, r, *_ = env.step(IDLE)
        total += r
    # potential-based: the sum depends only on the endpoints, so circling
    # without coming home pays (and costs) nothing net over a revolution
    expected = env.descent_reward * (d0 - env._pad_distance()) / (2.0 * p.radius)
    assert total == pytest.approx(expected, abs=1e-6)


def test_landing_on_pad_bonus_and_relabel():
    # generous crash_speed so an unpowered drop still counts as a landing
    env = OrbitAndReturnEnv(crash_speed=100.0)
    p = env.planet
    make_returning(env, 0.0, p.radius + 150.0, 0.0, -1.0)
    total, info = 0.0, None
    for _ in range(30):
        _, r, terminated, *_ , info = env.step(IDLE)
        total += r
        if terminated:
            break
    assert info["termination"] == "landed_on_pad"
    assert info["pad_downrange"] <= env.pad_radius
    assert total == pytest.approx(env.land_bonus, rel=0.01)


def test_landing_far_from_pad_gets_no_bonus():
    env = OrbitAndReturnEnv(crash_speed=100.0)
    p = env.planet
    ang = math.pi / 2.0 - 200_000.0 / p.radius  # 200 km east of the pad
    r = p.radius + 150.0
    make_returning(env, r * math.cos(ang), r * math.sin(ang), 0.0, 0.0)
    info = None
    for _ in range(30):
        _, r_, terminated, *_ , info = env.step(IDLE)
        if terminated:
            break
    assert info["termination"] == "landed"
    assert info["pad_downrange"] == pytest.approx(200_000.0, rel=0.01)


def test_crash_penalty_applies_in_return_phase():
    env = OrbitAndReturnEnv()
    p = env.planet
    make_returning(env, 0.0, p.radius + 3_000.0, 0.0, -200.0)
    total, info = 0.0, None
    for _ in range(60):
        _, r, terminated, *_ , info = env.step(IDLE)
        total += r
        if terminated:
            break
    assert info["termination"] == "crashed"
    assert total < -0.9 * env.crash_penalty


def test_return_autopilot_lands_on_the_pad():
    from scripts.autopilot_demo import ReturnAutopilot, fly

    env = OrbitAndReturnEnv()
    total_reward, info = fly(env, verbose=False, pilot=ReturnAutopilot(env))
    assert info["termination"] == "landed_on_pad"
    assert info["pad_downrange"] <= env.pad_radius
    assert info["dv_used"] < env.total_delta_v()
    assert total_reward > 500.0
