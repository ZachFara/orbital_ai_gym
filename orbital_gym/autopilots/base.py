"""Shared helpers for the scripted pilots."""

from __future__ import annotations

import math

import numpy as np


def throttle_action(env, throttle: float) -> float:
    """Map a desired throttle to the a[0] the env expects. With hover_control
    the action is hover-relative (throttle = hover * exp(a[0]*K)), so invert
    that; otherwise it is the plain [0,1] -> [-1,1]. Shared by every autopilot."""
    if getattr(env, "hover_control", False):
        hover = max(1e-4, env.hover_throttle())
        a0 = math.log(max(1e-4, throttle) / hover) / env.HOVER_K
        return float(np.clip(a0, -1.0, 1.0))
    return 2.0 * throttle - 1.0
