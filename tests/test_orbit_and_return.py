import numpy as np
import pytest

from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv

IDLE = np.array([-1.0, 0.0], dtype=np.float32)  # throttle 0, no turn
SPIN = np.array([-1.0, 1.0], dtype=np.float32)  # throttle 0, full turn


def test_return_mission_inherits_turn_penalty():
    env = OrbitAndReturnEnv()
    env.reset()
    total = 0.0
    for _ in range(20):
        _, r, *_ = env.step(SPIN)
        total += r
    assert total == pytest.approx(-20.0 * env.turn_penalty * env.step_duration, rel=1e-6)


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

    # Elliptic but still stable orbit at the same point (apoapsis):
    # lower tangential speed -> lower circularity -> slower hold progress.
    env.reset()
    env._teleport(0.0, r, 0.9 * v_circ, 0.0)
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
