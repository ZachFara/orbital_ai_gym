"""orbital_gym: 2D rocket mission environments for Gymnasium.

import gymnasium as gym
import orbital_gym

env = gym.make("TwoStageToOrbit-v0")           # launch to a stable orbit
env = gym.make("OrbitAndReturn-v0")            # ...then come home to the pad
env = gym.make("ToMoon-v0")                    # orbit, hold, land on the moon
env = gym.make("ToMars-v0")                    # three stages, a star, a Mars
env = gym.make("ToMoon-v1")                    # ...on legs: contact limits, 5 s hold
env = gym.make("TwoStageToOrbit-v0", planet="earth", max_episode_steps=8000)

The practice ladder: a mission's hardest leg on its own, from a cached
expert state, with difficulty knobs and `info["success"]`:

env = gym.make("MoonLandingFromOrbit-v0")             # parking orbit -> site
env = gym.make("MoonLandingFromOrbit-v0", level=0.3)  # start closer to touchdown
env = gym.make("MarsLandingFromOrbit-v0")             # the same leg at Mars
"""

from gymnasium.envs.registration import register

from .missions.orbit_and_return import OrbitAndReturnEnv
from .missions.to_mars import ToMarsEnv
from .missions.to_moon import ToMoonEnv
from .missions.to_orbit import TwoStageToOrbitEnv
from .presets import (
    MOONS,
    PLANETS,
    ROCKETS,
    STARS,
    SYSTEMS,
    Moon,
    OrbitingBody,
    Planet,
    Rocket,
    Stage,
    Star,
    StarSystem,
    describe,
    total_delta_v,
)
from .touchdown import TouchdownRules

__version__ = "0.1.0"
__all__ = [
    "MOONS",
    "PLANETS",
    "ROCKETS",
    "STARS",
    "SYSTEMS",
    "Moon",
    "OrbitAndReturnEnv",
    "OrbitingBody",
    "Planet",
    "Rocket",
    "Stage",
    "Star",
    "StarSystem",
    "ToMarsEnv",
    "ToMoonEnv",
    "TouchdownRules",
    "TwoStageToOrbitEnv",
    "describe",
    "total_delta_v",
]

# The envs enforce their own max_episode_steps (needed for internal reward
# bookkeeping), so no TimeLimit wrapper here.
register(
    id="TwoStageToOrbit-v0",
    entry_point="orbital_gym.missions.to_orbit:TwoStageToOrbitEnv",
    max_episode_steps=None,
)
register(
    id="OrbitAndReturn-v0",
    entry_point="orbital_gym.missions.orbit_and_return:OrbitAndReturnEnv",
    max_episode_steps=None,
)
register(
    id="ToMoon-v0",
    entry_point="orbital_gym.missions.to_moon:ToMoonEnv",
    max_episode_steps=None,
)
register(
    id="ToMars-v0",
    entry_point="orbital_gym.missions.to_mars:ToMarsEnv",
    max_episode_steps=None,
)
# v1 landing rules (decision 0010): legs with contact limits and a hold
register(
    id="ToMoon-v1",
    entry_point="orbital_gym.missions.to_moon:ToMoonEnv",
    max_episode_steps=None,
    kwargs={"touchdown": TouchdownRules()},
)
# The practice ladder (decision 0008): one leg of a mission, from a cached
# expert state. The wrapper enforces its own step window.
register(
    id="MoonLandingFromOrbit-v0",
    entry_point="orbital_gym.rl.landing:MoonLandingFromOrbit",
    max_episode_steps=None,
)
register(
    id="MarsLandingFromOrbit-v0",
    entry_point="orbital_gym.rl.landing:MarsLandingFromOrbit",
    max_episode_steps=None,
)
register(
    id="MoonLandingFromOrbit-v1",
    entry_point="orbital_gym.rl.landing:MoonLandingFromOrbitV1",
    max_episode_steps=None,
)
