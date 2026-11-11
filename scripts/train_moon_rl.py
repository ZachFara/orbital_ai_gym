"""PPO with a reachable-state curriculum for the lunar landing.

    python scripts/train_moon_rl.py --name ppo-curriculum-v1 --steps 20000000 \
        --hypothesis "..."
    python scripts/train_moon_rl.py --eval 32 --model models/<run>/final.zip
    python scripts/train_moon_rl.py --watch models/<run>/final.zip

Everything is logged to MLflow (scripts/explog.py): params, per-update
training stats, curriculum level, periodic deterministic evals from the real
start (parking orbit, random site), and the outcome/diagnosis at the end.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import explog
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from torch.nn import functional as F

from orbital_gym.rl.moon_landing import BANK_PATH, MoonLandingEnv, RewardConfig, load_bank


class BoundedPPO(PPO):
    """PPO with (a) a penalty on the Gaussian mean leaving the action box and
    (b) a floor on the exploration std.

    Why: the retro burn is a bang-bang decision (engine off in orbit, full
    thrust for 20 s, off again). With an unbounded Gaussian mean, "off" is an
    absorbing state: v11's mean throttle drifted to -2.1 with std 0.08 at the
    deorbit states, 13 sigma from the engine-on threshold, so no sampled
    action ever fired the engine and the gradient toward burning vanished.
    Keeping the mean inside [-1, 1] and the std above a floor keeps that
    gradient alive. (The penalty is the standard 'action bound loss'.)
    """

    bound_coef = 1.0
    std_floor = 0.2
    # optional behavior-cloning anchor: a FIXED set of expert (obs, action)
    # pairs (the state bank), regressed by the policy mean every minibatch.
    # Fixed, not the rollout buffer: anchoring to the buffer lets a drifting
    # policy anchor to its own drift (the TD3+BC lesson in train_her.py).
    bc_coef = 0.0
    bc_obs = None
    bc_act = None
    bc_batch = 256

    def set_bc_anchor(self, obs: np.ndarray, act: np.ndarray, coef: float) -> None:
        self.bc_obs = th.as_tensor(obs, device=self.device)
        self.bc_act = th.as_tensor(np.clip(act, -0.999, 0.999), device=self.device)
        self.bc_coef = coef

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, bound_losses, clip_fractions = [], [], [], [], []
        bc_losses = []
        continue_training = True
        for _epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                dist = self.policy.get_distribution(rollout_data.observations)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()
                mean = dist.distribution.mean
                values = self.policy.predict_values(rollout_data.observations).flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio, advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                bound_loss = th.mean(F.relu(mean.abs() - 1.0) ** 2)
                bound_losses.append(bound_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.bound_coef * bound_loss
                )
                if self.bc_coef > 0.0 and self.bc_obs is not None:
                    idx = th.randint(0, len(self.bc_act), (self.bc_batch,), device=self.device)
                    bc_mean = self.policy.get_distribution(self.bc_obs[idx]).distribution.mean
                    bc_loss = F.mse_loss(bc_mean, self.bc_act[idx])
                    bc_losses.append(bc_loss.item())
                    loss = loss + self.bc_coef * bc_loss
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                if self.target_kl is not None and approx_kl_divs[-1] > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                with th.no_grad():
                    self.policy.log_std.data.clamp_(min=math.log(self.std_floor))
            self._n_updates += 1
            if not continue_training:
                break
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bound_loss", np.mean(bound_losses))
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
            ),
        )
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def make_env(
    level,
    rc,
    gamma,
    seed,
    window,
    state_noise,
    site_noise,
    tmap,
    slack=400,
    cap_mult=1.0,
    extra_obs=False,
    site_sigma_max=0.04,
):
    def _thunk():
        return MoonLandingEnv(
            BANK_PATH,
            level=level,
            reward=rc,
            gamma=gamma,
            window=window,
            state_noise=state_noise,
            site_noise=site_noise,
            seed=seed,
            throttle_map=tmap,
            window_slack=slack,
            cap_mult=cap_mult,
            extra_obs=extra_obs,
            site_sigma_max=site_sigma_max,
        )

    return _thunk


class Curriculum(BaseCallback):
    """Raise the start-state level when the recent on-site rate is high enough;
    log episode statistics and training diagnostics to MLflow."""

    def __init__(
        self,
        level0: float,
        step: float,
        threshold: float,
        min_episodes: int,
        window: int = 200,
        log_every: int = 1,
        max_level: float = 1.0,
        probe_env=None,
        probe_episodes: int = 48,
        probe_every: int = 3,
    ):
        super().__init__()
        self.level = level0
        self.max_level = max_level
        # deterministic probe: the stochastic rollouts the gate used to read
        # run 10-15 points below the policy's real success rate (v4 @0.45:
        # 62% vs 74%), so the gate measures the mean policy at the current
        # level instead, once the noisy rate clears half the threshold
        self.probe_env = probe_env
        self.probe_episodes = probe_episodes
        self.probe_every = (
            probe_every  # rollouts between probes (each costs ~1 rollout of env steps)
        )
        self.probe_rate = float("nan")
        self.step_size = step
        self.threshold = threshold
        self.min_episodes = min_episodes
        self.win = window
        self.log_every = log_every
        self.recent: list[dict] = []
        self.at_level = 0
        self._rollouts = 0
        self.level_history: list[tuple[int, float]] = []

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self.training_env.env_method("set_level", self.level)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.recent.append(
                    {
                        "on_site": float(info["on_site"]),
                        "soft": float(info["soft"]),
                        "crash": float(info["termination"] == "crashed"),
                        "timeout": float(info["termination"] in ("window_limit", "time_limit")),
                        "escape": float(info["termination"] == "escaped"),
                        "miss_km": info["miss_km"],
                        "rel": info["moon_rel_speed"],
                        "ret": info["episode"]["r"],
                        "len": info["episode"]["l"],
                        "native": info["native_return"],
                        "dv": info["dv_spent"],
                        "ttl": info["start_ttl"],
                    }
                )
                self.at_level += 1
        if len(self.recent) > self.win:
            del self.recent[: -self.win]
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if not self.recent:
            return
        n = min(len(self.recent), self.at_level) or 1
        rec = self.recent[-n:]
        rate = float(np.mean([r["on_site"] for r in rec]))
        if (
            self.probe_env is not None
            and self.at_level >= self.min_episodes
            and rate >= 0.5 * self.threshold
            and self.level < self.max_level
            and self._rollouts % self.probe_every == 0
        ):
            self.probe_env.set_level(self.level)
            hits = 0
            for _ in range(self.probe_episodes):
                obs, _ = self.probe_env.reset()
                done = False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = self.probe_env.step(a)
                    done = term or trunc
                hits += int(info["on_site"])
            self.probe_rate = hits / self.probe_episodes
            rate = self.probe_rate
        elif self.probe_env is not None:
            rate = -1.0  # no fresh probe this rollout: never advance on the noisy rate
        if (
            self.at_level >= self.min_episodes
            and rate >= self.threshold
            and self.level < self.max_level
        ):
            self.probe_rate = float("nan")
            self.level = min(self.max_level, self.level + self.step_size)
            self.training_env.env_method("set_level", self.level)
            self.at_level = 0
            self.level_history.append((self.num_timesteps, self.level))
            print(f"  [{self.num_timesteps:>11,}] on-site {rate:.0%} -> level {self.level:.2f}")
        if self._rollouts % self.log_every == 0:
            m = {f"train/{k}": float(np.mean([r[k] for r in self.recent])) for k in self.recent[0]}
            m["train/level"] = self.level
            m["train/episodes_at_level"] = self.at_level
            if self.probe_rate == self.probe_rate:
                m["train/probe_on_site"] = self.probe_rate
            for k in (
                "train/approx_kl",
                "train/entropy_loss",
                "train/value_loss",
                "train/policy_gradient_loss",
                "train/explained_variance",
                "train/std",
                "train/bound_loss",
                "train/bc_loss",
            ):
                v = self.logger.name_to_value.get(k)
                if v is not None:
                    m[k.replace("train/", "ppo/")] = float(v)
            m["time/fps"] = self.num_timesteps / max(1e-9, time.time() - self._t0)
            import mlflow

            mlflow.log_metrics(m, step=self.num_timesteps)


class PeriodicEval(BaseCallback):
    def __init__(
        self,
        every: int,
        episodes: int,
        out_dir: str,
        rc,
        gamma,
        window,
        tmap: str = "exp",
        extra_obs: bool = False,
    ):
        super().__init__()
        self.every = every
        self.episodes = episodes
        self.out_dir = out_dir
        self._next = every
        self.env = MoonLandingEnv(
            BANK_PATH,
            nominal=True,
            reward=rc,
            gamma=gamma,
            window=window,
            state_noise=0.0,
            seed=12345,
            throttle_map=tmap,
            extra_obs=extra_obs,
        )
        self.best = -1.0
        self.last: dict = {}

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.last = evaluate(self.model, self.env, self.episodes, verbose=False)
            import mlflow

            mlflow.log_metrics(
                {f"eval/{k}": v for k, v in self.last.items()}, step=self.num_timesteps
            )
            print(
                f"  [eval @ {self.num_timesteps:>11,}] on-site {self.last['on_site']:.0%} "
                f"soft {self.last['soft']:.0%} crash {self.last['crash']:.0%} "
                f"miss {self.last['miss_km_med']:.1f} km  native {self.last['native']:.0f}"
            )
            self.model.save(os.path.join(self.out_dir, "last"))
            self.model.get_vec_normalize_env().save(os.path.join(self.out_dir, "last_vecnorm.pkl"))
            score = self.last["on_site"] + 0.1 * self.last["soft"]
            if score > self.best:
                self.best = score
                self.model.save(os.path.join(self.out_dir, "best"))
                self.model.get_vec_normalize_env().save(
                    os.path.join(self.out_dir, "best_vecnorm.pkl")
                )
        return True


def evaluate(model, env: MoonLandingEnv, episodes: int, verbose: bool = True, sites=None) -> dict:
    """Deterministic rollouts from the parking-orbit entry over a site grid."""
    sites = sites if sites is not None else np.linspace(-math.pi, math.pi, episodes, endpoint=False)
    rows = []
    for site in sites:
        obs, info = env.reset(options={"site_angle": float(site)})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        rows.append(info)
        if verbose:
            print(
                f"  site {math.degrees(site):+7.1f}: {info['termination']:<16} "
                f"rel {info['moon_rel_speed']:6.1f} m/s  miss {info['miss_km']:6.1f} km  "
                f"dv {info['dv_spent']:5.0f}  native {info['native_return']:6.0f}  "
                f"full-mission {info['native_return'] + env.bank.prefix_return:6.0f}"
            )
    out = {
        "on_site": np.mean([r["on_site"] for r in rows]),
        "soft": np.mean([r["soft"] for r in rows]),
        "crash": np.mean([r["termination"] == "crashed" for r in rows]),
        "timeout": np.mean([r["termination"] in ("window_limit", "time_limit") for r in rows]),
        "miss_km_med": float(np.median([r["miss_km"] for r in rows])),
        "rel_med": float(np.median([r["moon_rel_speed"] for r in rows])),
        "dv_mean": float(np.mean([r["dv_spent"] for r in rows])),
        "native": float(np.mean([r["native_return"] for r in rows])),
        "full_mission": float(np.mean([r["native_return"] for r in rows]) + env.bank.prefix_return),
    }
    if verbose:
        print(
            f"  -> on-site {out['on_site']:.0%}  soft {out['soft']:.0%}  crash {out['crash']:.0%}  "
            f"median miss {out['miss_km_med']:.1f} km  mean full-mission return {out['full_mission']:.0f}"
        )
    return {k: float(v) for k, v in out.items()}


def bc_dataset(bank, rc, gamma, window, tmap, stride: int = 1, extra_obs: bool = False):
    """(obs, action) pairs for every bank state, in the wrapper's spaces."""
    from orbital_gym.rl.moon_landing import restore

    env = MoonLandingEnv(
        BANK_PATH,
        level=1.0,
        reward=rc,
        gamma=gamma,
        window=window,
        state_noise=0.0,
        throttle_map=tmap,
        extra_obs=extra_obs,
    )
    obs, act = [], []
    for e in bank.entries[::stride]:
        restore(env.env, e.snap)
        env.env.set_site_angle(e.site)
        env.start_ttl = e.ttl
        env._t = 0
        env._cap = min(window, e.ttl + 400)
        obs.append(env._obs())
        act.append(env.expert_action(e))
    return np.stack(obs), np.stack(act)


def bc_pretrain(
    model: PPO,
    obs: np.ndarray,
    act: np.ndarray,
    epochs: int,
    batch: int = 512,
    lr: float = 3e-4,
    log_std: float = -1.0,
) -> None:
    """Regress the policy mean onto the expert's from-orbit actions (warm start)."""
    policy = model.policy
    policy.set_training_mode(True)
    obs_t = th.as_tensor(obs, device=policy.device)
    act_t = th.as_tensor(np.clip(act, -0.999, 0.999), device=policy.device)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_t)
    for epoch in range(epochs):
        perm = th.randperm(n, device=policy.device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            dist = policy.get_distribution(obs_t[idx])
            loss = th.nn.functional.mse_loss(dist.distribution.mean, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"      bc epoch {epoch + 1}/{epochs}: mse {tot / n:.5f}")
    with th.no_grad():
        policy.log_std.data.fill_(log_std)
    policy.set_training_mode(False)


def load_model(path: str, env_fn):
    """Load a saved PPO with its VecNormalize stats around a fresh eval env."""
    base = path[:-4] if path.endswith(".zip") else path
    stats = base + "_vecnorm.pkl"
    vec = DummyVecEnv([env_fn])
    if os.path.exists(stats):
        vec = VecNormalize.load(stats, vec)
        vec.training = False
        vec.norm_reward = False
    return PPO.load(path, device="cpu"), vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument(
        "--bound-coef",
        type=float,
        default=0.0,
        help="penalty on the policy mean outside [-1,1] (0 = plain PPO)",
    )
    ap.add_argument(
        "--std-floor",
        type=float,
        default=0.0,
        help="minimum exploration std (0 = none); needs --bound-coef > 0",
    )
    ap.add_argument("--net", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--norm-obs", action="store_true")
    ap.add_argument("--level0", type=float, default=0.0)
    ap.add_argument("--max-level", type=float, default=1.0, help="curriculum ceiling (ablations)")
    ap.add_argument("--throttle-map", default="exp", choices=["lin", "exp"])
    ap.add_argument("--window-slack", type=int, default=400)
    ap.add_argument(
        "--extra-obs",
        action="store_true",
        help="append zero-effort miss/velocity guidance channels to the observation",
    )
    ap.add_argument("--cap-mult", type=float, default=1.0)
    ap.add_argument("--sde", action="store_true", help="gSDE: smooth, state-dependent exploration")
    ap.add_argument("--sde-freq", type=int, default=8)
    ap.add_argument("--level-step", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-episodes", type=int, default=150)
    ap.add_argument("--probe-every", type=int, default=3, help="rollouts between probes")
    ap.add_argument(
        "--site-sigma-max",
        type=float,
        default=0.04,
        help="cap on descent-start site offset noise, rad",
    )
    ap.add_argument(
        "--probe-episodes",
        type=int,
        default=48,
        help="deterministic episodes at the current level that gate a level "
        "advance (0 = gate on the stochastic rollouts)",
    )
    ap.add_argument("--window", type=int, default=1800)
    ap.add_argument("--state-noise", type=float, default=1.0)
    ap.add_argument("--site-noise", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # reward knobs
    for f in RewardConfig.__dataclass_fields__.values():
        ap.add_argument(f"--r-{f.name.replace('_', '-')}", type=float, default=f.default)
    ap.add_argument(
        "--bc-coef",
        type=float,
        default=0.0,
        help="weight of a behavior-cloning anchor to the expert bank during PPO "
        "(needs --bound-coef > 0)",
    )
    ap.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="warm-start: behavior-clone the policy mean onto the expert's "
        "from-orbit actions for this many epochs before PPO",
    )
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--watch", default=None)
    args = ap.parse_args()

    rc = RewardConfig(**{f: getattr(args, f"r_{f}") for f in RewardConfig.__dataclass_fields__})

    if args.eval or args.watch:
        path = args.model or args.watch
        env = MoonLandingEnv(
            BANK_PATH,
            nominal=True,
            reward=rc,
            gamma=args.gamma,
            window=args.window,
            state_noise=0.0,
            seed=12345,
            throttle_map=args.throttle_map,
            extra_obs=args.extra_obs,
            render_mode="human" if args.watch else None,
        )
        model, vec = load_model(path, lambda: env)
        if isinstance(vec, VecNormalize) and vec.norm_obs:
            raise SystemExit("norm-obs models need the vec env; not supported in eval yet")
        evaluate(model, env, args.eval or 8)
        return

    print("[1/3] expert state bank")
    bank = load_bank()
    print(
        f"      {len(bank.entries)} states, descent {bank.descent_len} steps, "
        f"max ttl {bank.max_ttl}, prefix return {bank.prefix_return:.0f}"
    )

    name = args.name or time.strftime("ppo_%Y%m%d_%H%M%S")
    out_dir = os.path.join("models", name)
    os.makedirs(out_dir, exist_ok=True)
    params = {
        **vars(args),
        "bank_states": len(bank.entries),
        "descent_len": bank.descent_len,
        "max_ttl": bank.max_ttl,
    }
    explog.start(name, params, args.hypothesis, tags={"algo": "PPO", "task": "orbit->site"})

    print("[2/3] building vec env + PPO")
    vec = SubprocVecEnv(
        [
            make_env(
                args.level0,
                rc,
                args.gamma,
                args.seed * 1000 + i,
                args.window,
                args.state_noise,
                args.site_noise,
                args.throttle_map,
                args.window_slack,
                args.cap_mult,
                args.extra_obs,
                args.site_sigma_max,
            )
            for i in range(args.envs)
        ]
    )
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=args.norm_obs, norm_reward=True, gamma=args.gamma)
    policy_kwargs = dict(
        net_arch=dict(pi=list(args.net), vf=list(args.net)), log_std_init=args.log_std_init
    )
    algo = PPO
    if args.bound_coef > 0.0:
        algo = BoundedPPO
        BoundedPPO.bound_coef = args.bound_coef
        BoundedPPO.std_floor = max(1e-3, args.std_floor)
    if args.resume:
        stats = (
            args.resume[:-4] + "_vecnorm.pkl"
            if args.resume.endswith(".zip")
            else args.resume + "_vecnorm.pkl"
        )
        if os.path.exists(stats):
            vec = VecNormalize.load(stats, vec.venv)
            vec.training = True
            print(f"      restored reward-normalisation stats from {stats}")
        model = algo.load(args.resume, env=vec, device="cpu", learning_rate=args.lr)
        if args.bound_coef > 0.0:
            with th.no_grad():
                model.policy.log_std.data.clamp_(min=math.log(BoundedPPO.std_floor))
    else:
        model = algo(
            "MlpPolicy",
            vec,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device="cpu",
            verbose=0,
            use_sde=args.sde,
            sde_sample_freq=args.sde_freq if args.sde else -1,
        )
    if args.bc_epochs > 0 or args.bc_coef > 0.0:
        bo, ba = bc_dataset(
            bank, rc, args.gamma, args.window, args.throttle_map, extra_obs=args.extra_obs
        )
        if args.bc_epochs > 0 and not args.resume:
            print(f"      behavior cloning on {len(bo)} expert states, {args.bc_epochs} epochs")
            bc_pretrain(model, bo, ba, args.bc_epochs, log_std=args.log_std_init)
        if args.bc_coef > 0.0:
            model.set_bc_anchor(bo, ba, args.bc_coef)
            print(f"      BC anchor on {len(bo)} expert pairs, coef {args.bc_coef}")
    probe_env = None
    if args.probe_episodes > 0:
        probe_env = MoonLandingEnv(
            BANK_PATH,
            level=args.level0,
            reward=rc,
            gamma=args.gamma,
            window=args.window,
            state_noise=args.state_noise,
            site_noise=args.site_noise,
            seed=777,
            throttle_map=args.throttle_map,
            window_slack=args.window_slack,
            cap_mult=args.cap_mult,
            extra_obs=args.extra_obs,
            site_sigma_max=args.site_sigma_max,
        )
    cur = Curriculum(
        args.level0,
        args.level_step,
        args.threshold,
        args.min_episodes,
        max_level=args.max_level,
        probe_env=probe_env,
        probe_episodes=args.probe_episodes,
        probe_every=args.probe_every,
    )
    ev = PeriodicEval(
        args.eval_every,
        args.eval_episodes,
        out_dir,
        rc,
        args.gamma,
        args.window,
        tmap=args.throttle_map,
        extra_obs=args.extra_obs,
    )

    print(f"[3/3] training {args.steps:,} steps -> {out_dir}")
    t0 = time.time()
    status, outcome, diagnosis = "FINISHED", "", ""
    try:
        model.learn(total_timesteps=args.steps, callback=[cur, ev])
    except KeyboardInterrupt:
        status = "KILLED"
        outcome = "interrupted"
    finally:
        model.save(os.path.join(out_dir, "final"))
        vec.save(os.path.join(out_dir, "final_vecnorm.pkl"))
        print("final evaluation (parking orbit, 32 sites):")
        final = evaluate(model, ev.env, 32)
        import mlflow

        mlflow.log_metrics({f"final/{k}": v for k, v in final.items()}, step=model.num_timesteps)
        mlflow.log_artifacts(out_dir)
        mins = (time.time() - t0) / 60
        outcome = outcome or (
            f"level reached {cur.level:.2f}; final on-site {final['on_site']:.0%}, "
            f"soft {final['soft']:.0%}, crash {final['crash']:.0%}, "
            f"median miss {final['miss_km_med']:.1f} km, "
            f"{model.num_timesteps:,} steps in {mins:.0f} min"
        )
        diagnosis = diagnosis or (
            "(fill in with `mlflow` tag edit or explog) see curriculum "
            "level history: "
            + ", ".join(f"{s // 1000}k:{lvl:.2f}" for s, lvl in cur.level_history[-12:])
        )
        explog.finish(
            outcome,
            diagnosis,
            status,
            summary_metrics={
                "final/level": cur.level,
                **{f"final/{k}": v for k, v in final.items()},
            },
        )
        print(f"saved {out_dir}/final.zip; {outcome}")


if __name__ == "__main__":
    main()
