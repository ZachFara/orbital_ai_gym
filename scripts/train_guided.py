"""Guided PPO: use the scripted autopilots as the starting point of training.

Plain PPO takes millions of steps to (maybe) discover these missions from
scratch; the scripted autopilots already fly them. This script turns each
autopilot into three kinds of head start:

1. Demonstrations -- roll the autopilot with small exploration noise on the
   *executed* actions while recording its *clean* commands, so the dataset
   covers a tube around the expert trajectory with corrective labels.
2. Behavior cloning -- regress the policy's action mean onto the
   demonstrations: the network starts as a neural copy of the autopilot.
3. Jump-start curriculum -- every training episode begins with the autopilot
   flying a random prefix of the mission (replayed from a cached action
   sequence) before the policy takes over. The maximum prefix anneals to
   zero, so the policy learns the end of the mission first and works its way
   back to the launch pad (JSRL-style).

The moon mission is the motivating case: its autopilot brakes to a slow
near-touchdown but runs the tank dry a few km up -- the policy starts from
the autopilot's skills and only has to find the last ~100 m/s of efficiency.

    pip install -e ".[train]"
    python scripts/train_guided.py --mission orbit|return|moon   # 1M steps
    python scripts/train_guided.py --mission moon --resume models/<run>.zip
    python scripts/train_guided.py --mission moon --eval 5 --watch models/<run>.zip
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
except ImportError:
    raise SystemExit("stable-baselines3 not installed; run: pip install -e '.[train]'") from None

import gymnasium as gym

from orbital_gym.autopilots import Autopilot, MoonAutopilot, ReturnAutopilot
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv

MISSIONS = {
    "orbit": (TwoStageToOrbitEnv, Autopilot),
    "return": (OrbitAndReturnEnv, ReturnAutopilot),
    "moon": (ToMoonEnv, MoonAutopilot),
}

GAMMA = 0.999  # long horizon: the mission rewards arrive late


def expert_actions(mission: str) -> list[np.ndarray]:
    """One clean expert episode, cached as its action sequence.

    The env has no reset randomness and the autopilots are deterministic,
    so replaying these actions reproduces the expert trajectory exactly.
    """
    env_cls, pilot_cls = MISSIONS[mission]
    env = env_cls()
    _obs, info = env.reset()
    pilot = pilot_cls(env)
    actions = []
    while True:
        a = pilot(info)
        actions.append(np.asarray(a, dtype=np.float32))
        _obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    env.close()
    return actions


class JumpStart(gym.Wrapper):
    """Autopilot flies a random prefix of each episode; the policy finishes.

    The prefix is replayed from the cached expert actions, and its length is
    sampled uniformly in [prefix_min, prefix_max] * len(expert episode).
    With prefix_min = 0 and the ceiling annealed to zero this is a reverse
    curriculum over the whole mission; with a window like [0.88, 0.97] (the
    moon default) every episode starts on final approach and the policy
    only ever trains the landing -- the one phase the autopilot hasn't
    mastered. The env state at prefix_min is snapshotted so resets replay
    only the window tail, not the whole flight.
    """

    SNAP_SKIP = ("_renderer",)  # never snapshot these env fields

    def __init__(
        self, env: gym.Env, actions: list[np.ndarray], prefix_min: float, prefix_max: float
    ):
        super().__init__(env)
        self._actions = actions
        self.prefix_min = prefix_min
        self.prefix_max = prefix_max
        self._rng = np.random.default_rng()
        self._snap = None  # env state at prefix_min, so resets replay only the tail

    def set_prefix_max(self, value: float) -> None:
        self.prefix_max = max(float(value), self.prefix_min)

    def _snapshot(self):
        u = self.env.unwrapped
        return {
            k: copy.deepcopy(v)
            for k, v in u.__dict__.items()
            if k.startswith("_") and k not in self.SNAP_SKIP
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_min = int(self.prefix_min * len(self._actions))
        n_max = int(self.prefix_max * len(self._actions))
        if n_max <= 0:
            return obs, info
        if n_min > 0:
            if self._snap is None:
                for i in range(n_min):
                    self.env.step(self._actions[i])
                self._snap = self._snapshot()
            else:
                self.env.unwrapped.__dict__.update(copy.deepcopy(self._snap))
        n = int(self._rng.integers(n_min, n_max + 1))
        for i in range(n_min, n):
            _, _, term, trunc, _ = self.env.step(self._actions[i])
            if term or trunc:  # expert episode ended inside the prefix
                if self._snap is not None:
                    self.env.unwrapped.__dict__.update(copy.deepcopy(self._snap))
                else:
                    self.env.reset(**kwargs)
                break
        u = self.env.unwrapped
        return u._obs(), u._info()


def make_env(mission: str, actions: list[np.ndarray], prefix_min: float, prefix_max: float):
    def _thunk():
        return JumpStart(MISSIONS[mission][0](), actions, prefix_min, prefix_max)

    return _thunk


class LiveProgress(BaseCallback):
    """Something to watch while it trains.

    Every plot_every steps, print an ASCII graph of mean episode return
    over training so far. Every probe_every steps, fly one deterministic
    episode with the current policy (autopilot prefix included) and print
    how it went; when it beats the best return so far, replay it in the
    pygame window -- in a separate process, so training keeps running
    while you watch. The env is deterministic, so the recorded actions
    are the whole flight; the prefix is fast-forwarded invisibly and what
    you watch is the part the policy flies.
    """

    def __init__(
        self,
        mission: str,
        actions: list[np.ndarray],
        handoff: float,
        plot_every: int = 20_000,
        probe_every: int = 50_000,
        render: bool = True,
    ):
        super().__init__()
        self.mission = mission
        self._actions = actions
        self._n_pre = int(handoff * len(actions))
        self.plot_every = plot_every
        self.probe_every = probe_every
        self.render = render
        self._returns: list[tuple[int, float]] = []
        self._best = -math.inf
        self._next_plot = plot_every
        self._next_probe = probe_every
        self._probe_env = None
        self._proc = None  # the detached replay-window process

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                self._returns.append((self.num_timesteps, float(ep["r"])))
        if self.plot_every and self.num_timesteps >= self._next_plot:
            self._next_plot += self.plot_every
            self._plot()
        if self.probe_every and self.num_timesteps >= self._next_probe:
            self._next_probe += self.probe_every
            self._probe()
        return True

    def _plot(self, width: int = 60, height: int = 10) -> None:
        if len(self._returns) < 2:
            return
        t_max = self._returns[-1][0]
        sums = np.zeros(width)
        counts = np.zeros(width)
        for t, r in self._returns:
            c = min(width - 1, int(t / t_max * width))
            sums[c] += r
            counts[c] += 1
        cols = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        filled = cols[~np.isnan(cols)]
        lo, hi = float(filled.min()), float(filled.max())
        span = max(1e-9, hi - lo)
        print(
            f"\n  episode return over training  "
            f"(mean of last 20: {np.mean([r for _, r in self._returns[-20:]]):.0f}, "
            f"best probe: {self._best if self._best > -1e9 else float('nan'):.0f})"
        )
        for row in range(height, 0, -1):
            label = f"{lo + span * row / height:8.0f} |" if row in (height, 1) else "         |"
            line = "".join(
                " " if np.isnan(c) else ("█" if (c - lo) / span * height >= row - 0.5 else " ")
                for c in cols
            )
            print(f"  {label}{line}")
        print(f"          +{'-' * width}\n           0 steps{' ' * (width - 18)}{t_max:,}\n")

    def _probe(self) -> None:
        if self._probe_env is None:
            self._probe_env = MISSIONS[self.mission][0]()
        ret, info, acts = self._fly(self._probe_env)
        line = (
            f"  [probe @ {self.num_timesteps:>9,}] end={info['termination']:<16} return={ret:7.1f}"
        )
        if self.mission == "moon":
            env = self._probe_env
            line += (
                f"  moon alt {(info['moon_distance'] - env.moon.radius) / 1000:7.1f} km"
                f"  rel {info['moon_rel_speed']:5.0f} m/s"
            )
        if ret > self._best:
            self._best = ret
            line += "  NEW BEST"
            print(line)
            if self.render:
                print("  replaying it in a window; training continues...")
                self._spawn_replay(acts)
        else:
            print(line)

    def _fly(self, env):
        obs, info = env.reset()
        acts = []
        done = False
        for i in range(self._n_pre):
            acts.append(self._actions[i])
            obs, _, term, trunc, info = env.step(self._actions[i])
            done = term or trunc
            if done:
                break
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            acts.append(action)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        return info["episode_return"], info, acts

    def _spawn_replay(self, acts: list[np.ndarray]) -> None:
        """Replay the flight in a detached process so training never waits
        for the window. A newer best supersedes a still-open window."""
        path = os.path.join(
            tempfile.gettempdir(), f"guided_replay_{os.getpid()}_{self.num_timesteps}.npz"
        )
        np.savez(path, actions=np.stack(acts), n_pre=self._n_pre, mission=self.mission)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--replay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class AnnealPrefix(BaseCallback):
    """Linearly anneal the max jump-start prefix down to the window floor
    over `frac` of training."""

    def __init__(self, prefix0: float, prefix_min: float, frac: float, total_steps: int):
        super().__init__()
        self.prefix0 = prefix0
        self.prefix_min = prefix_min
        self.end = max(1, int(frac * total_steps))

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            f = max(0.0, 1.0 - self.num_timesteps / self.end)
            p = self.prefix_min + (self.prefix0 - self.prefix_min) * f
            self.training_env.env_method("set_prefix_max", p)
        return True


def collect_demos(mission: str, episodes: int, noise: float, seed: int = 0):
    """DART-style demos: noisy execution, clean labels."""
    env_cls, pilot_cls = MISSIONS[mission]
    rng = np.random.default_rng(seed)
    all_obs, all_act = [], []
    for ep in range(episodes):
        env = env_cls()
        obs, info = env.reset()
        pilot = pilot_cls(env)
        sigma = 0.0 if ep == 0 else noise  # first episode is the clean one
        n = 0
        while True:
            a = np.asarray(pilot(info), dtype=np.float32)
            all_obs.append(np.asarray(obs, dtype=np.float32))
            all_act.append(a)
            exec_a = np.clip(a + rng.normal(0.0, sigma, a.shape), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(exec_a.astype(np.float32))
            n += 1
            if term or trunc:
                break
        print(
            f"  demo ep {ep + 1}/{episodes}: {n} steps, sigma={sigma:.2f}, "
            f"end={info['termination']}, return={info['episode_return']:.0f}"
        )
        env.close()
    return np.stack(all_obs), np.stack(all_act)


def bc_pretrain(
    model: PPO, obs: np.ndarray, act: np.ndarray, epochs: int, batch: int = 512, lr: float = 3e-4
) -> None:
    """Regress the policy's action mean onto the expert actions."""
    policy = model.policy
    policy.set_training_mode(True)
    device = policy.device
    obs_t = th.as_tensor(obs, device=device)
    act_t = th.as_tensor(act, device=device)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_t)
    for epoch in range(epochs):
        perm = th.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            dist = policy.get_distribution(obs_t[idx])
            loss = th.nn.functional.mse_loss(dist.distribution.mean, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  bc epoch {epoch + 1}/{epochs}: mse {total / n:.5f}")
    # BC sets the action mean; without this the Gaussian keeps std=1.0 and
    # on-policy rollouts are near-random on [-1,1] actions no matter how
    # good the clone is
    with th.no_grad():
        policy.log_std.data.fill_(math.log(0.12))
    policy.set_training_mode(False)


def rollout_check(model: PPO, mission: str) -> None:
    """One deterministic rollout of the current policy, before PPO."""
    env = MISSIONS[mission][0]()
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(
        f"      bc policy solo: end={info['termination']}, "
        f"return={info['episode_return']:.0f}, dv={info['dv_used']:.0f}"
    )
    env.close()


def replay(path: str) -> None:
    """Play back a recorded flight in the pygame window. Spawned as its own
    process by LiveProgress so training never blocks on the replay; the
    autopilot prefix is stepped with rendering off, then the window opens
    at the handoff and the policy's part plays in real time."""
    import pygame

    data = np.load(path)
    actions, n_pre, mission = data["actions"], int(data["n_pre"]), str(data["mission"])
    with contextlib.suppress(OSError):
        os.remove(path)
    env = MISSIONS[mission][0]()
    env.reset()
    for i in range(n_pre):
        _, _, term, trunc, _ = env.step(actions[i])
        if term or trunc:
            env.close()
            return
    env.render_mode = "human"
    for a in actions[n_pre:]:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc or pygame.event.peek(pygame.QUIT):
            break
    for _ in range(120):  # hold the outcome banner ~4 s, then close
        if pygame.event.peek(pygame.QUIT):
            break
        env.render()
    env.close()


def evaluate(
    model_path: str, episodes: int, device: str, mission: str, handoff: float = 0.0
) -> None:
    """Roll the policy; with handoff > 0, the autopilot flies that fraction
    of its recorded episode first (the hybrid the landing window trains)."""
    env = MISSIONS[mission][0]()
    model = PPO.load(model_path, device=device)
    prefix = expert_actions(mission) if handoff > 0.0 else []
    n_pre = int(handoff * len(prefix))
    successes = softs = 0
    returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        for i in range(n_pre):
            obs, _, terminated, truncated, info = env.step(prefix[i])
            done = terminated or truncated
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        returns.append(info["episode_return"])
        if mission == "return":
            success = info["termination"] == "landed_on_pad"
            extra = f"touchdown {info['pad_downrange'] / 1000:7.2f} km from pad"
        elif mission == "moon":
            success = info["termination"] in ("landed_moon_site", "landed_moon")
            mvx, mvy = env._moon_velocity()
            rel = math.hypot(env._vx - mvx, env._vy - mvy)
            # the env labels a soft off-site touchdown "crashed"; count it
            on_moon = info["moon_distance"] <= env.moon.radius * 1.001
            soft = on_moon and rel <= env.moon_landing_speed
            softs += soft
            extra = (
                f"moon alt {(info['moon_distance'] - env.moon.radius) / 1000:7.1f} km, "
                f"rel {rel:5.0f} m/s, site {info['moon_site_distance'] / 1000:6.0f} km"
                f"{'  SOFT TOUCHDOWN' if soft and not success else ''}"
            )
        else:
            success = info["termination"] == "time_limit" and info["orbit"]
            extra = f"e {info['eccentricity']:5.3f}"
        successes += success
        print(
            f"ep {ep + 1:2d}: {info['termination']:<16} "
            f"return {info['episode_return']:7.1f}  "
            f"dv used {info['dv_used']:4.0f} m/s  {extra}"
            f"{'  SUCCESS' if success else ''}"
        )
    print("-" * 64)
    line = f"succeeded {successes}/{episodes}"
    if mission == "moon":
        line += f"  soft moon touchdowns {softs}/{episodes}"
    print(f"{line}  mean return {sum(returns) / len(returns):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="moon", choices=sorted(MISSIONS))
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--demos", type=int, default=8, help="demo episodes for BC")
    ap.add_argument("--noise", type=float, default=0.05, help="demo action noise")
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument(
        "--prefix",
        type=float,
        default=None,
        help="initial max jump-start prefix, fraction of the expert "
        "episode (default: 0.97 moon, 0.85 others)",
    )
    ap.add_argument(
        "--prefix-min",
        type=float,
        default=None,
        help="jump-start window floor: episodes never start earlier "
        "than this fraction. The moon default (0.88) pins "
        "training to the landing -- the one phase the "
        "autopilot hasn't mastered (default: 0 others)",
    )
    ap.add_argument(
        "--anneal",
        type=float,
        default=0.7,
        help="anneal the prefix ceiling to the floor over this fraction of training",
    )
    ap.add_argument(
        "--lr", type=float, default=1e-4, help="PPO learning rate (kept low to protect the BC init)"
    )
    ap.add_argument(
        "--plot-every",
        type=int,
        default=20_000,
        help="print the reward-over-training graph every N steps (0=off)",
    )
    ap.add_argument(
        "--probe-every",
        type=int,
        default=50_000,
        help="test-fly the current policy every N steps (0=off)",
    )
    ap.add_argument(
        "--no-render",
        action="store_true",
        help="don't open the window that replays new best flights",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--resume",
        default=None,
        metavar="MODEL.zip",
        help="continue training an existing model for --steps more "
        "steps (skips demos and behavior cloning; restores "
        "reward-normalization stats from MODEL_vecnorm.pkl "
        "when present)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--watch", default=None)
    # internal: replay a flight recorded by LiveProgress (own process)
    ap.add_argument("--replay", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.replay:
        replay(args.replay)
        return
    if args.prefix is None:
        args.prefix = 0.97 if args.mission == "moon" else 0.85
    if args.prefix_min is None:
        args.prefix_min = 0.88 if args.mission == "moon" else 0.0
    # fractions of the expert episode; accept "90" as a friendly 90%
    args.prefix = min(args.prefix / 100.0 if args.prefix > 1.0 else args.prefix, 0.995)
    args.prefix_min = min(
        args.prefix_min / 100.0 if args.prefix_min > 1.0 else args.prefix_min, args.prefix
    )

    if args.eval:
        if not (args.watch or args.out):
            raise SystemExit("--eval needs a model: --watch models/<run>.zip")
        evaluate(
            args.watch or args.out, args.eval, args.device, args.mission, handoff=args.prefix_min
        )
        return

    if args.watch:
        import pygame

        env = MISSIONS[args.mission][0](render_mode="human")
        model = PPO.load(args.watch, device=args.device)
        prefix = expert_actions(args.mission) if args.prefix_min > 0.0 else []
        n_pre = int(args.prefix_min * len(prefix))
        obs, info = env.reset()
        env.render()
        pre_i = 0
        while True:
            if pygame.event.peek(pygame.QUIT):
                break
            if pre_i < n_pre:  # autopilot flies the approach
                action = prefix[pre_i]
                pre_i += 1
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"{info['termination']}  return {info['episode_return']:.1f}")
                obs, info = env.reset()
                pre_i = 0
        env.close()
        return

    if args.resume:
        if not args.resume.endswith(".zip"):
            args.resume += ".zip"
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume: no such model: {args.resume}")
        print(f"[1/3] resuming from {args.resume} -- skipping demos and BC")
    else:
        print(f"[1/3] collecting {args.demos} demo episodes ({args.mission})")
        demo_obs, demo_act = collect_demos(args.mission, args.demos, args.noise)
        print(f"      {len(demo_obs)} state-action pairs")

    print("[2/3] caching the expert episode for jump-starts")
    actions = expert_actions(args.mission)
    print(f"      expert episode: {len(actions)} steps")

    inner = VecMonitor(
        SubprocVecEnv(
            [
                make_env(args.mission, actions, args.prefix_min, args.prefix)
                for _ in range(args.envs)
            ]
        )
    )
    # The value function is trained against VecNormalize's running reward
    # scale, so a resumed run must restore the stats saved with the model.
    stats = args.resume[:-4] + "_vecnorm.pkl" if args.resume else None
    if stats and os.path.exists(stats):
        vec = VecNormalize.load(stats, inner)
        vec.training = True
        print(f"      restored reward-normalization stats from {stats}")
    else:
        vec = VecNormalize(inner, norm_obs=False, norm_reward=True, gamma=GAMMA)
        if args.resume:
            print(
                "      no saved reward-normalization stats found; "
                "starting them fresh (value estimates settle in ~50k steps)"
            )

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec,
            device=args.device,
            learning_rate=args.lr,
            verbose=0 if args.plot_every else 1,
        )
        print(
            f"      model had {model.num_timesteps:,} training steps; training {args.steps:,} more"
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=256,
            gamma=GAMMA,
            gae_lambda=0.98,
            device=args.device,
            seed=0,
            verbose=0 if args.plot_every else 1,
        )
        print(f"[2/3] behavior cloning, {args.bc_epochs} epochs")
        bc_pretrain(model, demo_obs, demo_act, args.bc_epochs)
        rollout_check(model, args.mission)

    print(
        f"[3/3] jump-start PPO, {args.steps} steps, "
        f"prefix {args.prefix} -> {args.prefix_min} over {args.anneal:.0%}"
    )
    out = args.out or time.strftime(f"models/guided_{args.mission}_%Y%m%d_%H%M%S")
    callbacks = [AnnealPrefix(args.prefix, args.prefix_min, args.anneal, args.steps)]
    if args.plot_every or args.probe_every:
        callbacks.append(
            LiveProgress(
                args.mission,
                actions,
                handoff=args.prefix_min,
                plot_every=args.plot_every,
                probe_every=args.probe_every,
                render=not args.no_render,
            )
        )
    model.learn(total_timesteps=args.steps, callback=callbacks)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    vec.save((out[:-4] if out.endswith(".zip") else out) + "_vecnorm.pkl")
    print(
        f"saved {out}.zip -- evaluate with:\n"
        f"  python scripts/train_guided.py --mission {args.mission} "
        f"--eval 5 --watch {out}.zip\n"
        f"train it further with:\n"
        f"  python scripts/train_guided.py --mission {args.mission} "
        f"--resume {out}.zip"
    )


if __name__ == "__main__":
    main()
