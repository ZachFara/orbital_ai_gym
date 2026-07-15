"""orbital_gym: 2D rocket mission environments for Gymnasium.

    import gymnasium as gym
    import orbital_gym

    env = gym.make("TwoStageToOrbit-v0")           # launch to a stable orbit
    env = gym.make("OrbitAndReturn-v0")            # ...then come home to the pad
    env = gym.make("TwoStageToOrbit-v0", planet="earth", max_episode_steps=8000)
"""

from gymnasium.envs.registration import register

from .presets import MOONS, PLANETS, ROCKETS, Moon, Planet, Rocket, Stage, describe, total_delta_v
from .missions.to_orbit import TwoStageToOrbitEnv
from .missions.orbit_and_return import OrbitAndReturnEnv
from .missions.to_moon import ToMoonEnv

__version__ = "0.1.0"
__all__ = [
    "TwoStageToOrbitEnv",
    "OrbitAndReturnEnv",
    "ToMoonEnv",
    "Moon",
    "Planet",
    "Rocket",
    "Stage",
    "MOONS",
    "PLANETS",
    "ROCKETS",
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
