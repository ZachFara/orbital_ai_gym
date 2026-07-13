"""orbital_gym: 2D two-stage-to-orbit Gymnasium environment.

    import gymnasium as gym
    import orbital_gym

    env = gym.make("TwoStageToOrbit-v0")           # small "easy" planet
    env = gym.make("TwoStageToOrbit-v0", planet="earth", max_episode_steps=8000)
"""

from gymnasium.envs.registration import register

from .presets import PLANETS, ROCKETS, Planet, Rocket, Stage, describe, total_delta_v
from .two_stage_env import TwoStageToOrbitEnv

__version__ = "0.1.0"
__all__ = [
    "TwoStageToOrbitEnv",
    "Planet",
    "Rocket",
    "Stage",
    "PLANETS",
    "ROCKETS",
    "describe",
    "total_delta_v",
]

register(
    id="TwoStageToOrbit-v0",
    entry_point="orbital_gym.two_stage_env:TwoStageToOrbitEnv",
    # The env enforces its own max_episode_steps (needed for the orbit bonus),
    # so no TimeLimit wrapper here.
    max_episode_steps=None,
)
