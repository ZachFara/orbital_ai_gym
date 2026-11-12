"""Scripted reference pilots, one class per mission.

They prove each mission closes, serve as the baseline in the README tables
and generate the start states the RL tasks train from. A learned policy
never needs them at test time.
"""

from .base import throttle_action
from .lambert import lambert
from .mars import MarsAutopilot
from .moon import MoonAutopilot
from .orbit import Autopilot, ReturnAutopilot

__all__ = [
    "Autopilot",
    "MarsAutopilot",
    "MoonAutopilot",
    "ReturnAutopilot",
    "lambert",
    "throttle_action",
]
