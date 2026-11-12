"""The benchmark protocol: a fixed set of episodes per registered env, three
baselines (random, the scripted autopilot, a trained policy), one table.

Episodes are fixed so numbers are comparable across commits (decision 0009):
missions with a landing site fly an even grid of sites, the others a run of
seeds; the practice ids fly their evaluation distribution (`nominal=True`,
parking-orbit entry) over a 16-site grid. `python scripts/benchmark.py`
reproduces the table in the README.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .autopilots import Autopilot, MarsAutopilot, MoonAutopilot, ReturnAutopilot

# the most capable scripted pilot per mission (the moon's orbit-first pilot
# lands on the site; the direct-descent one is a frozen baseline for the
# preset site only)
PILOTS: dict[str, Callable[[Any], Any]] = {
    "TwoStageToOrbit-v0": Autopilot,
    "OrbitAndReturn-v0": ReturnAutopilot,
    "ToMoon-v0": lambda env: MoonAutopilot(env, orbit=True),
    "ToMars-v0": MarsAutopilot,
}
SITE_MISSIONS = ("ToMoon-v0", "ToMars-v0")
PRACTICE_IDS = ("MoonLandingFromOrbit-v0", "MarsLandingFromOrbit-v0")
ENV_IDS = (*PILOTS, *PRACTICE_IDS)
POLICIES = ("random", "autopilot", "ppo")


@dataclass(frozen=True)
class Episode:
    env_id: str
    seed: int
    site: float | None  # site angle (rad), for the envs that have one


def site_grid(n: int) -> list[float]:
    return [float(s) for s in np.linspace(-math.pi, math.pi, n, endpoint=False)]


def protocol(env_id: str, episodes: int | None = None) -> list[Episode]:
    """The fixed episodes of one env (`episodes` overrides the count)."""
    if env_id in PRACTICE_IDS:
        sites = site_grid(episodes or 16)
        return [Episode(env_id, 1000 + i, s) for i, s in enumerate(sites)]
    if env_id in SITE_MISSIONS:
        sites = site_grid(episodes or 8)
        return [Episode(env_id, i, s) for i, s in enumerate(sites)]
    return [Episode(env_id, i, None) for i in range(episodes or 8)]


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env[Any, Any]:
    if env_id in PRACTICE_IDS:
        return gym.make(env_id, nominal=True, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)


# --- policies ----------------------------------------------------------------------------


class Policy(Protocol):
    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray: ...


def random_policy(env: gym.Env[Any, Any], seed: int) -> Policy:
    env.action_space.seed(seed)

    def act(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return np.asarray(env.action_space.sample(), dtype=np.float32)

    return act


def autopilot_policy(env: gym.Env[Any, Any], env_id: str) -> Policy:
    """The mission's scripted pilot; for a practice id, the task's orbit-first
    expert started in its parking-orbit mode, its env-frame actions mapped
    into the wrapper's body-frame action space."""
    u: Any = env.unwrapped
    if env_id not in PRACTICE_IDS:
        pilot = PILOTS[env_id](u)

        def act(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
            return np.asarray(pilot(info), dtype=np.float32)

        return act

    from .rl.landing import BankEntry

    expert = u.task.make_pilot(u.env)
    expert.mode = "moon_orbit"  # the pilots' name for "holding the parking orbit"

    def act_from_orbit(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        a_env = np.asarray(expert(info), dtype=np.float32)
        out: np.ndarray = u.expert_action(BankEntry({}, 0, 0.0, expert.mode, act=a_env))
        return out

    return act_from_orbit


def ppo_policy(path: str) -> Policy:
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")

    def act(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return act


# --- running ----------------------------------------------------------------------------


@dataclass
class Outcome:
    termination: str
    success: bool
    ret: float
    steps: int
    dv_used: float


@dataclass
class Recorder:
    """Keeps every `every`-th rendered frame, downscaled by `scale`, at most
    `max_frames` of them (uniformly thinned)."""

    every: int = 8
    scale: int = 2
    max_frames: int = 240
    frames: list[np.ndarray] | None = None

    def capture(self, env: gym.Env[Any, Any], step: int) -> None:
        if step % self.every:
            return
        frame = env.render()
        if frame is None:
            return
        if self.frames is None:
            self.frames = []
        self.frames.append(np.asarray(frame)[:: self.scale, :: self.scale])

    def save(self, path: str, fps: int = 20) -> int:
        import imageio

        frames: list[Any] = list(self.frames or [])
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
        return len(frames)


def run_episode(
    env: gym.Env[Any, Any], policy: Policy, ep: Episode, recorder: Recorder | None = None
) -> Outcome:
    options = {"site_angle": ep.site} if ep.site is not None else None
    obs, info = env.reset(seed=ep.seed, options=options)
    ret, steps, done = 0.0, 0, False
    if recorder is not None:
        recorder.capture(env, 0)
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs, info))
        ret += float(reward)
        steps += 1
        done = terminated or truncated
        if recorder is not None:
            recorder.capture(env, steps)
    return Outcome(
        str(info["termination"]), bool(info["success"]), ret, steps, float(info["dv_used"])
    )


@dataclass
class Row:
    env_id: str
    policy: str
    episodes: int
    success: float
    ret: float
    dv_used: float
    steps: float
    seconds: float
    terminations: dict[str, int]


def make_policy(
    name: str, env: gym.Env[Any, Any], env_id: str, seed: int, models: dict[str, str]
) -> Policy | None:
    if name == "random":
        return random_policy(env, seed)
    if name == "autopilot":
        return autopilot_policy(env, env_id)
    if name == "ppo":
        return ppo_policy(models[env_id]) if env_id in models else None
    raise ValueError(name)


def run(
    env_id: str,
    policy_name: str,
    models: dict[str, str] | None = None,
    episodes: int | None = None,
    record: str | None = None,
) -> Row | None:
    """Every protocol episode of one env under one baseline; None when the
    baseline does not exist for it (no model given). `record` saves a GIF
    of the first episode to that path."""
    models = models or {}
    eps = protocol(env_id, episodes)
    outs: list[Outcome] = []
    t0 = time.time()
    for i, ep in enumerate(eps):
        recording = record is not None and i == 0
        env = make_env(env_id, render_mode="rgb_array" if recording else None)
        policy = make_policy(policy_name, env, env_id, ep.seed, models)
        if policy is None:
            env.close()
            return None
        recorder = Recorder() if recording else None
        outs.append(run_episode(env, policy, ep, recorder))
        if recorder is not None and record is not None:
            recorder.save(record)
        env.close()
    terms: dict[str, int] = {}
    for o in outs:
        terms[o.termination] = terms.get(o.termination, 0) + 1
    return Row(
        env_id,
        policy_name,
        len(outs),
        float(np.mean([o.success for o in outs])),
        float(np.mean([o.ret for o in outs])),
        float(np.mean([o.dv_used for o in outs])),
        float(np.mean([o.steps for o in outs])),
        time.time() - t0,
        terms,
    )


def table(rows: Iterable[Row]) -> str:
    """The rows as a Markdown table."""
    lines = [
        "| env | policy | episodes | success | mean return | mean dv (m/s) | mean steps | outcomes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        outcomes = ", ".join(
            f"{k} {v}" for k, v in sorted(r.terminations.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"| `{r.env_id}` | {r.policy} | {r.episodes} | {r.success:.0%} | {r.ret:.0f} | "
            f"{r.dv_used:.0f} | {r.steps:.0f} | {outcomes} |"
        )
    return "\n".join(lines)
