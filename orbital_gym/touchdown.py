"""Landing legs: the touchdown rules of the `-v1` landing envs (decision 0010).

Under v0 a touchdown is one number: body-relative speed under a limit at the
instant of contact, and the episode ends there. Under v1 the lander has legs
and a flight computer that cares about more than the speed: contact must be
slow vertically, slow along the surface and nearly upright, and the episode
ends only after the vehicle has sat on the surface for a hold with the engine
effectively off. Firing the engine hard enough to lift off resets the hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TouchdownRules:
    max_vertical: float = 5.0  # m/s downward at contact
    max_horizontal: float = 3.0  # m/s along the surface at contact
    max_tilt_deg: float = 15.0  # degrees off the local vertical at contact
    hold_seconds: float = 5.0  # on the surface with the engine effectively off
    engine_off_fraction: float = 0.1  # thrust below this fraction of local weight is "off"

    def violated(self, vr: float, vt: float, tilt: float) -> str | None:
        """The first limit a contact at radial speed `vr` (negative = down),
        tangential speed `vt` and tilt (rad, off the local vertical) breaks,
        or None for a touchdown the legs take."""
        if -vr > self.max_vertical:
            return "vertical"
        if abs(vt) > self.max_horizontal:
            return "horizontal"
        if abs(tilt) > math.radians(self.max_tilt_deg):
            return "tilt"
        return None
