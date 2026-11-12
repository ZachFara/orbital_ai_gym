"""The headless renderer draws every mission, including the body-relative
views (moon sphere of influence, star-system view, landed on Mars)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from orbital_gym.autopilots import MarsAutopilot, MoonAutopilot
from orbital_gym.missions.to_mars import ToMarsEnv
from orbital_gym.missions.to_moon import ToMoonEnv

W, H = 960, 720


def _frame(env: Any, settle: int = 1) -> np.ndarray:
    """Render `settle` frames (the camera eases between framings) and return
    the last one, checked for shape and for not being blank."""
    frame = None
    for _ in range(settle):
        frame = env.render()
    assert frame is not None
    assert frame.shape == (H, W, 3)
    assert frame.mean() > 2.0  # not a blank surface
    return np.asarray(frame)


def _fly_to(env: Any, pilot: Any, stop: Any) -> dict[str, Any]:
    """Keep flying the already-running mission until `stop(info)` holds."""
    info = env.unwrapped._info()
    while not stop(info):
        _, _, terminated, truncated, info = env.step(pilot(info))
        if terminated or truncated:
            break
    return dict(info)


@pytest.mark.parametrize("site", [0.0, math.pi])
def test_moon_mission_renders_in_every_frame_of_reference(site: float) -> None:
    env = ToMoonEnv(render_mode="rgb_array")
    env.reset(options={"site_angle": site})
    pilot = MoonAutopilot(env)
    _frame(env)  # on the pad
    info = _fly_to(env, pilot, lambda i: i["orbit_progress"] >= 1.0)
    _frame(env, settle=30)  # parking orbit: map view around the planet
    info = _fly_to(env, pilot, lambda i: i["soi"] == "moon")
    assert info["soi"] == "moon"
    _frame(env, settle=30)  # inside the moon's sphere: re-centred on the moon


def test_mars_mission_renders_the_system_view_and_the_target() -> None:
    env = ToMarsEnv(render_mode="rgb_array")
    env.reset()
    pilot = MarsAutopilot(env)
    _frame(env)
    info = _fly_to(env, pilot, lambda i: i["soi"] == "sun")
    assert info["soi"] == "sun"
    frame = _frame(env, settle=120)  # star-system view: the star sits at the centre
    assert frame[H // 2 - 40 : H // 2 + 40, W // 2 - 40 : W // 2 + 40].mean() > 20.0
    info = _fly_to(env, pilot, lambda i: i["soi"] == "mars")
    assert info["soi"] == "mars"
    _frame(env, settle=30)  # Mars-centred
