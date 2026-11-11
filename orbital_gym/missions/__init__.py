"""Mission environments, one module per registered env ID.

    to_orbit.py          TwoStageToOrbit-v0   launch into a stable orbit
    orbit_and_return.py  OrbitAndReturn-v0    orbit half a revolution, then
                                              come home to the launch pad
    to_moon.py           ToMoon-v0            orbit, hold, then transfer to a moon site
    to_mars.py           ToMars-v0            three stages: escape the planet, cross the
                                              star system, land on a Mars-like target
"""

from .to_orbit import TwoStageToOrbitEnv
from .orbit_and_return import OrbitAndReturnEnv
from .to_moon import ToMoonEnv
from .to_mars import ToMarsEnv

__all__ = ["TwoStageToOrbitEnv", "OrbitAndReturnEnv", "ToMoonEnv", "ToMarsEnv"]
