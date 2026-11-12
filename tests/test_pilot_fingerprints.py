"""Exact fingerprints of every scripted flight.

The pilots are reference results and start-state generators, so a refactor
must leave their flights bit-for-bit identical. These numbers were recorded
before the pilots were restructured onto the `Pilot` base class; a change
here is either a deliberate pilot change (update the numbers in the same
PR and say so) or a bug.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from orbital_gym.autopilots import Autopilot, MarsAutopilot, MoonAutopilot, ReturnAutopilot
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_mars import ToMarsEnv
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv

Flight = tuple[Callable[[], Any], Callable[[Any], Any], int, str, float, float]

FLIGHTS: dict[str, Flight] = {
    # name: (env factory, pilot factory, steps, termination, dv_used, episode return)
    "orbit": (TwoStageToOrbitEnv, Autopilot, 4000, "time_limit", 4265.609, 7459.778),
    "return": (OrbitAndReturnEnv, ReturnAutopilot, 2212, "landed_on_pad", 5217.490, 601.288),
    "moon-direct": (ToMoonEnv, MoonAutopilot, 2956, "landed_moon", 6522.727, 2792.753),
    "moon-orbit-first": (
        lambda: ToMoonEnv(max_episode_steps=12_000, hover_control=True),
        lambda env: MoonAutopilot(env, orbit=True),
        6149,
        "landed_moon_site",
        8387.510,
        3089.210,
    ),
    "mars": (ToMarsEnv, MarsAutopilot, 15296, "landed_mars", 8563.426, 2125.663),
}


@pytest.mark.parametrize("name", sorted(FLIGHTS))
def test_scripted_flight_is_unchanged(name: str) -> None:
    make_env, make_pilot, steps, termination, dv_used, episode_return = FLIGHTS[name]
    env = make_env()
    _, info = env.reset()
    pilot = make_pilot(env)
    n = 0
    while True:
        _, _, terminated, truncated, info = env.step(pilot(info))
        n += 1
        if terminated or truncated:
            break
    assert (n, info["termination"]) == (steps, termination)
    assert info["dv_used"] == pytest.approx(dv_used, abs=1e-3)
    assert info["episode_return"] == pytest.approx(episode_return, abs=1e-3)
