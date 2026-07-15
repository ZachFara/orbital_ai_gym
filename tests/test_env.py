import math

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import orbital_gym
from orbital_gym.presets import G0, PLANETS, ROCKETS, stage_delta_v, total_delta_v
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv

IDLE = np.array([-1.0, 0.0], dtype=np.float32)   # throttle 0, no turn
BURN = np.array([1.0, 0.0], dtype=np.float32)    # full throttle, hold attitude


def test_gymnasium_api_conformance():
    check_env(TwoStageToOrbitEnv(), skip_render_check=True)


def test_registered_id_and_kwargs():
    env = gym.make("TwoStageToOrbit-v0", planet="earth")
    assert env.unwrapped.planet.name == "earth"
    env.close()


def test_sits_on_pad_without_thrust():
    env = TwoStageToOrbitEnv()
    env.reset()
    total = 0.0
    for _ in range(30):
        _, r, terminated, truncated, info = env.step(IDLE)
        total += r
        assert not terminated and not truncated
    assert info["altitude"] == pytest.approx(0.0, abs=1.0)
    assert total == pytest.approx(0.0, abs=1e-6)


def test_vertical_climb_reward_matches_altitude():
    env = TwoStageToOrbitEnv()
    _, info = env.reset()
    total = 0.0
    for _ in range(60):
        _, r, *_ , info = env.step(BURN)
        total += r
    assert info["altitude"] > 3_000.0
    expected = env.climb_reward * min(info["altitude"], env.planet.space_altitude) \
        / env.planet.space_altitude
    assert total == pytest.approx(expected, rel=1e-6)


def test_auto_staging_drops_dry_mass():
    env = TwoStageToOrbitEnv()
    env.reset()
    s1 = env.rocket.stages[0]
    burn_time = s1.prop_mass * s1.isp * G0 / s1.thrust
    info = None
    for _ in range(int(burn_time) + 5):
        *_, info = env.step(BURN)
        if info["stage"] == 1:
            break
    assert info["stage"] == 1
    upper = env.rocket.stages[1]
    assert info["mass"] <= upper.dry_mass + upper.prop_mass + 1e-6


def test_tsiolkovsky_presets():
    r = ROCKETS["easy"]
    m0 = r.launch_mass
    m1 = m0 - r.stages[0].prop_mass
    dv1 = r.stages[0].isp * G0 * math.log(m0 / m1)
    assert stage_delta_v(r, 0) == pytest.approx(dv1)
    assert total_delta_v(r) == pytest.approx(stage_delta_v(r, 0) + stage_delta_v(r, 1))
    env = TwoStageToOrbitEnv()
    env.reset()
    assert env.delta_v_remaining() == pytest.approx(total_delta_v(r), rel=1e-9)


def test_orbit_energy_conserved_without_thrust():
    env = TwoStageToOrbitEnv()
    env.reset()
    p = env.planet
    alt = 200_000.0
    v = p.circular_velocity(alt)
    env._teleport(0.0, p.radius + alt, v, 0.0)
    energy0 = env._orbit_elements()[2]
    for _ in range(1900):  # just over one full orbit
        _, r, terminated, truncated, info = env.step(IDLE)
        assert not terminated
    peri, apo, energy, ecc = env._orbit_elements()
    assert abs(energy - energy0) / abs(energy0) < 0.005
    assert peri == pytest.approx(alt, rel=0.02)
    assert apo == pytest.approx(alt, rel=0.02)
    assert info["orbit"] is True


def test_space_reward_is_time_above_cutoff():
    env = TwoStageToOrbitEnv(orbit_rate=0.0)  # isolate the space term
    env.reset()
    p = env.planet
    alt = 200_000.0
    env._teleport(0.0, p.radius + alt, p.circular_velocity(alt), 0.0)
    total = 0.0
    for _ in range(50):
        _, r, *_ = env.step(IDLE)
        total += r
    # 50 steps x 1 s x space_rate, no climb reward (already above the cutoff)
    assert total == pytest.approx(50.0 * env.space_rate * env.step_duration, rel=1e-6)


def test_circularity_reward_pays_for_round_orbits_only():
    env = TwoStageToOrbitEnv()
    env.reset()
    p = env.planet
    alt = 200_000.0

    # Circular orbit: space_rate plus nearly the full orbit_rate per second.
    env._teleport(0.0, p.radius + alt, p.circular_velocity(alt), 0.0)
    total, info = 0.0, None
    for _ in range(50):
        _, r, *_, info = env.step(IDLE)
        total += r
    assert info["orbit"] and info["circularity"] > 0.95
    expected = 50.0 * (env.space_rate + env.orbit_rate * info["circularity"])
    assert total == pytest.approx(expected * env.step_duration, rel=0.01)

    # The giant-ellipse exploit: in space, but periapsis buried in the planet.
    # Only the plain space reward accrues, so circularizing dominates it.
    env.reset()
    env._teleport(0.0, p.radius + alt, 0.4 * p.circular_velocity(alt), 100.0)
    total = 0.0
    for _ in range(30):
        _, r, *_, info = env.step(IDLE)
        total += r
    assert not info["orbit"] and info["periapsis"] < 0.0
    assert total == pytest.approx(30.0 * env.space_rate * env.step_duration, rel=1e-6)


def test_turn_penalty_discourages_spinning():
    env = TwoStageToOrbitEnv()
    env.reset()
    spin = np.array([-1.0, 1.0], dtype=np.float32)  # throttle 0, full turn
    total = 0.0
    for _ in range(20):
        _, r, *_ = env.step(spin)
        total += r
    assert total == pytest.approx(-20.0 * env.turn_penalty * env.step_duration, rel=1e-6)


def test_crash_penalty_and_termination():
    env = TwoStageToOrbitEnv()
    env.reset()
    p = env.planet
    env._teleport(0.0, p.radius + 3_000.0, 0.0, -200.0)
    total, terminated, info = 0.0, False, None
    for _ in range(60):
        _, r, terminated, truncated, info = env.step(IDLE)
        total += r
        if terminated:
            break
    assert terminated and info["termination"] == "crashed"
    assert total < -env.crash_penalty * 0.5  # penalty minus the altitude refund


def test_escape_terminates_when_out_of_fuel():
    env = TwoStageToOrbitEnv()
    env.reset()
    p = env.planet
    v_escape = math.sqrt(2 * p.mu / (p.radius + 200_000.0))
    env._teleport(0.0, p.radius + 200_000.0, 0.0, 1.2 * v_escape)
    env._stage = len(env.rocket.stages) - 1
    env._fuel = 0.0
    _, _, terminated, _, info = env.step(IDLE)
    assert terminated and info["termination"] == "escaped"


def test_terminate_on_orbit_bonus():
    env = TwoStageToOrbitEnv(terminate_on_orbit=True)
    env.reset()
    p = env.planet
    alt = 100_000.0
    env._teleport(0.0, p.radius + alt, p.circular_velocity(alt), 0.0)
    _, reward, terminated, _, info = env.step(IDLE)
    assert terminated and info["termination"] == "orbit"
    remaining = env.max_episode_steps - 1
    rate = env.space_rate + env.orbit_rate * info["circularity"]
    assert reward == pytest.approx(rate * env.step_duration * (remaining + 1), rel=0.01)


def test_rgb_array_render_headless():
    env = TwoStageToOrbitEnv(render_mode="rgb_array")
    env.reset()
    env.step(BURN)
    frame = env.render()
    assert frame.shape == (720, 960, 3) and frame.dtype == np.uint8
    env.close()


def test_autopilot_reaches_orbit():
    from scripts.autopilot_demo import fly

    env = TwoStageToOrbitEnv(terminate_on_orbit=True)
    total_reward, info = fly(env, verbose=False)
    assert info["termination"] == "orbit"
    assert info["periapsis"] > env.planet.space_altitude
    assert info["dv_used"] < env.total_delta_v()
    assert total_reward > 2_000.0
