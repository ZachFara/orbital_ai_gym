"""The v1 landing rules: each contact limit, the hold, the reset on
lift-off, the observation and info channels, v0 untouched, and the
orbit-first pilot still landing under them."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import orbital_gym  # noqa: F401
from orbital_gym.autopilots import MoonAutopilot
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.touchdown import TouchdownRules

OFF = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
RULES = TouchdownRules()


def _place(env: Any, altitude: float, vr: float, vt: float, tilt_deg: float = 0.0) -> None:
    """Hover the rocket `altitude` above the moon's near face with the given
    moon-relative radial/tangential speeds and tilt off the local vertical,
    the site right below it, on the last stage with some fuel."""
    body = env.target_body
    mx, my = body.position(env._time)
    mvx, mvy = body.velocity(env._time)
    d = body.radius + altitude
    ux, uy = mx / math.hypot(mx, my), my / math.hypot(mx, my)
    up = math.atan2(uy, ux)
    env._teleport(
        mx + ux * d,
        my + uy * d,
        mvx + ux * vr + uy * vt,
        mvy + uy * vr - ux * vt,
        theta=up + math.radians(tilt_deg),
    )
    env._target_theta = env._theta
    env._stage = len(env.rocket.stages) - 1
    stage = env.rocket.stages[-1]
    env._fuel = 0.5 * stage.prop_mass
    env._mass = stage.dry_mass + env._fuel
    env.set_site_angle(up - body.angle(env._time))


def _v1(**kw: Any) -> Any:
    env = ToMoonEnv(touchdown=RULES, hover_control=True, **kw)
    env.reset()
    return env


@pytest.mark.parametrize(
    ("vr", "vt", "tilt", "limit"),
    [(-6.0, 0.0, 0.0, "vertical"), (-1.0, 4.0, 0.0, "horizontal"), (-1.0, 0.0, 20.0, "tilt")],
)
def test_each_contact_limit_is_a_crash(vr: float, vt: float, tilt: float, limit: str) -> None:
    env = _v1()
    _place(env, 1.0, vr, vt, tilt)
    _, _, terminated, _, info = env.step(OFF)
    assert terminated and info["termination"] == "crashed"
    assert info["touchdown_limit"] == limit and info["success"] is False
    # the same contact is a landing under v0's single speed limit
    v0: Any = ToMoonEnv(hover_control=True)
    v0.reset()
    _place(v0, 1.0, vr, vt, tilt)
    _, _, terminated, _, info = v0.step(OFF)
    assert terminated and info["termination"].startswith("landed_moon")


def test_soft_contact_holds_five_seconds_and_lift_off_resets_the_clock() -> None:
    env = _v1()
    _place(env, 1.0, -1.0, 0.5, 5.0)
    obs, _, terminated, _, info = env.step(OFF)
    # the contact step itself counts: the engine was off for its whole second
    assert not terminated and info["contact"] and info["hold_progress"] == pytest.approx(0.2)
    assert obs.shape == (32,) and obs[-2] == 1.0 and obs[-1] == pytest.approx(0.2)
    assert info["moon_rel_speed"] < 1e-6 and info["moon_distance"] == pytest.approx(env.moon.radius)
    for k in (2, 3):
        obs, _, terminated, _, info = env.step(OFF)
        assert not terminated and info["hold_progress"] == pytest.approx(k / 5)
        assert obs[-1] == pytest.approx(k / 5)
    # the engine coming on (below lift-off thrust) starts the hold over
    idle = np.array([-0.4, 1.0, 0.0], dtype=np.float32)  # hover * exp(-1.2) ~ 0.3 g
    _, _, terminated, _, info = env.step(idle)
    assert not terminated and info["contact"] and info["hold_progress"] == 0.0
    # full thrust lifts off: contact lost, clock reset, the rocket is airborne
    _, _, terminated, _, info = env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert not terminated and not info["contact"] and info["hold_progress"] == 0.0
    assert info["moon_distance"] > env.moon.radius
    # a fresh soft contact and the full hold with the engine off: landed on
    # the site, five seconds after touching down
    _place(env, 1.0, -1.0, 0.0, 0.0)
    n = 0
    while True:
        _, reward, terminated, truncated, info = env.step(OFF)
        n += 1
        assert not truncated
        if terminated:
            break
    assert info["termination"] == "landed_moon_site" and info["success"]
    assert n == 5 and info["hold_progress"] == 1.0 and reward > env.moon_site_bonus - 1.0


def test_v0_is_untouched_and_the_ids_are_registered() -> None:
    v0 = ToMoonEnv()
    _, info = v0.reset()
    assert v0.observation_space.shape == (30,) and "contact" not in info
    v1 = gym.make("ToMoon-v1")
    assert v1.observation_space.shape == (32,)
    _, info = v1.reset()
    assert info["contact"] is False and info["hold_progress"] == 0.0
    practice: Any = gym.make("MoonLandingFromOrbit-v1").unwrapped
    assert practice.observation_space.shape[0] == 32 + 1 + 8
    practice.close()


def test_orbit_first_pilot_lands_on_the_legs() -> None:
    env: Any = ToMoonEnv(touchdown=RULES, max_episode_steps=12_000, hover_control=True)
    _, info = env.reset(options={"site_angle": 0.0})
    pilot = MoonAutopilot(env, orbit=True)
    held = 0
    while True:
        _, _, terminated, truncated, info = env.step(pilot(info))
        held += int(info["contact"])
        if terminated or truncated:
            break
    assert info["termination"] == "landed_moon_site" and info["success"]
    assert held >= 5 and info["hold_progress"] == 1.0
