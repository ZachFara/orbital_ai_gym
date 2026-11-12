"""Training the landing policy: configuration, the PPO variant, the
curriculum and evaluation callbacks, and the `TrainingRun` that wires them.

The defaults in `TrainConfig` are the recipe that produced the reference
result (run `ppo-zem-v13` from scratch, polished as `ppo-zem-v15`; see
EXPERIMENTS.md): exponential hover throttle, the zero-effort guidance
channels, a potential on the predicted miss, a cost for hovering near the
surface, gamma 0.9995, bounded policy means with an exploration floor, and
a behavior-cloning warm start plus a fixed-bank BC anchor. A bare
`python scripts/train_moon_rl.py --name x --hypothesis y` runs it.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from torch.nn import functional as F

from .moon_landing import BANK_PATH, MoonLandingEnv, RewardConfig, StateBank, load_bank, restore

Metrics = dict[str, float]


@dataclass
class TrainConfig:
    """Every knob of a training run, with the reference recipe as defaults."""

    name: str = ""
    hypothesis: str = ""
    steps: int = 30_000_000
    envs: int = 8
    n_steps: int = 1024
    batch_size: int = 2048
    epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.9995
    gae_lambda: float = 0.98
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = -1.0
    bound_coef: float = 1.0  # penalty on the policy mean outside [-1, 1]; 0 = plain PPO
    std_floor: float = 0.2  # minimum exploration std (with bound_coef > 0)
    net: tuple[int, ...] = (256, 256)
    norm_obs: bool = False
    level0: float = 0.1
    max_level: float = 1.0
    level_step: float = 0.05
    threshold: float = 0.8
    min_episodes: int = 150
    probe_episodes: int = 24  # deterministic episodes gating a level advance (0 = noisy gate)
    probe_every: int = 5  # rollouts between probes
    throttle_map: str = "exp"
    extra_obs: bool = True  # the zero-effort guidance channels
    window: int = 1800
    window_slack: int = 400
    cap_mult: float = 1.0
    state_noise: float = 1.0
    site_noise: float = 1.0
    site_sigma_max: float = 0.04
    sde: bool = False
    sde_freq: int = 8
    bc_epochs: int = 20  # behavior-cloning warm start on the expert bank
    bc_coef: float = 0.2  # fixed-bank BC anchor inside PPO
    eval_every: int = 500_000
    eval_episodes: int = 16
    seed: int = 0
    resume: str = ""  # checkpoint to continue from (its _vecnorm.pkl is picked up)
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            w_miss=2.0, fuel=0.0005, hover_cost=0.02, hover_alt=500.0
        )
    )

    def flat(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        reward = d.pop("reward")
        d.update({f"r_{k}": v for k, v in reward.items()})
        return d


# --- the PPO variant ------------------------------------------------------------------


def policy_mean(dist: Any) -> th.Tensor:
    """The mean of a (single, diagonal-Gaussian) SB3 action distribution."""
    d = dist.distribution
    assert not isinstance(d, list)
    mean: th.Tensor = d.mean
    return mean


def backward(loss: th.Tensor) -> None:
    loss.backward()  # type: ignore[no-untyped-call]  # torch ships it untyped


class BoundedPPO(PPO):
    """PPO with a penalty on the Gaussian mean leaving the action box, a floor
    on the exploration std, and an optional behavior-cloning anchor.

    The retro burn is a bang-bang decision. With an unbounded Gaussian mean,
    "engine off" is an absorbing state: a mean throttle of -2 with std 0.08
    never samples an engine-on action and the gradient toward burning
    vanishes (run v11). Bounding the mean and flooring the std keep it alive.
    The anchor regresses the mean toward a FIXED set of expert pairs (the
    state bank), never toward the rollout buffer, which would let a drifting
    policy anchor to its own drift. Without it a unimodal Gaussian never
    discovered the 20 s full burn at all (runs v10-v12).
    """

    bound_coef = 1.0
    std_floor = 0.2
    bc_coef = 0.0
    bc_batch = 256
    bc_obs: th.Tensor | None = None
    bc_act: th.Tensor | None = None

    def set_bc_anchor(self, obs: np.ndarray, act: np.ndarray, coef: float) -> None:
        self.bc_obs = th.as_tensor(obs, device=self.device)
        self.bc_act = th.as_tensor(np.clip(act, -0.999, 0.999), device=self.device)
        self.bc_coef = coef

    def _bc_loss(self) -> th.Tensor | None:
        if self.bc_coef <= 0.0 or self.bc_obs is None or self.bc_act is None:
            return None
        idx = th.randint(0, len(self.bc_act), (self.bc_batch,), device=self.device)
        mean = policy_mean(self.policy.get_distribution(self.bc_obs[idx]))
        return F.mse_loss(mean, self.bc_act[idx])

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        cr = self.clip_range
        clip_range = float(cr(self._current_progress_remaining) if callable(cr) else cr)
        logs: dict[str, list[float]] = {
            k: [] for k in ("entropy", "pg", "value", "bound", "bc", "clip_frac", "kl")
        }
        continue_training = True
        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                dist = self.policy.get_distribution(rollout_data.observations)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()
                mean = policy_mean(dist)
                values = self.policy.predict_values(rollout_data.observations).flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                bound_loss = th.mean(F.relu(mean.abs() - 1.0) ** 2)
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.bound_coef * bound_loss
                )
                bc_loss = self._bc_loss()
                if bc_loss is not None:
                    loss = loss + self.bc_coef * bc_loss
                    logs["bc"].append(bc_loss.item())
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    kl = float(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())
                logs["pg"].append(policy_loss.item())
                logs["value"].append(value_loss.item())
                logs["entropy"].append(entropy_loss.item())
                logs["bound"].append(bound_loss.item())
                logs["clip_frac"].append(float(th.mean((th.abs(ratio - 1) > clip_range).float())))
                logs["kl"].append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                backward(loss)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                with th.no_grad():
                    self.policy.log_std.data.clamp_(min=math.log(self.std_floor))
            self._n_updates += 1
            if not continue_training:
                break
        self.logger.record("train/entropy_loss", float(np.mean(logs["entropy"])))
        self.logger.record("train/policy_gradient_loss", float(np.mean(logs["pg"])))
        self.logger.record("train/value_loss", float(np.mean(logs["value"])))
        self.logger.record("train/bound_loss", float(np.mean(logs["bound"])))
        if logs["bc"]:
            self.logger.record("train/bc_loss", float(np.mean(logs["bc"])))
        self.logger.record("train/approx_kl", float(np.mean(logs["kl"])))
        self.logger.record("train/clip_fraction", float(np.mean(logs["clip_frac"])))
        self.logger.record(
            "train/explained_variance",
            float(
                explained_variance(
                    self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
                )
            ),
        )
        self.logger.record("train/std", float(th.exp(self.policy.log_std).mean().item()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


# --- environments -----------------------------------------------------------------------


def make_env(
    cfg: TrainConfig, seed: int, level: float | None = None, **overrides: Any
) -> Callable[[], MoonLandingEnv]:
    """A factory for one training env (SubprocVecEnv needs a picklable thunk)."""
    kwargs: dict[str, Any] = dict(
        level=cfg.level0 if level is None else level,
        reward=cfg.reward,
        gamma=cfg.gamma,
        window=cfg.window,
        state_noise=cfg.state_noise,
        site_noise=cfg.site_noise,
        seed=seed,
        throttle_map=cfg.throttle_map,
        window_slack=cfg.window_slack,
        cap_mult=cfg.cap_mult,
        extra_obs=cfg.extra_obs,
        site_sigma_max=cfg.site_sigma_max,
    )
    kwargs.update(overrides)

    def _thunk() -> MoonLandingEnv:
        return MoonLandingEnv(BANK_PATH, **kwargs)

    return _thunk


def nominal_env(
    cfg: TrainConfig, render_mode: str | None = None, seed: int = 12345
) -> MoonLandingEnv:
    """The evaluation env: every episode starts at parking-orbit entry."""
    return MoonLandingEnv(
        BANK_PATH,
        nominal=True,
        reward=cfg.reward,
        gamma=cfg.gamma,
        window=cfg.window,
        state_noise=0.0,
        seed=seed,
        throttle_map=cfg.throttle_map,
        extra_obs=cfg.extra_obs,
        render_mode=render_mode,
    )


# --- evaluation -------------------------------------------------------------------------


Step = tuple[np.ndarray, float, bool, bool, dict[str, Any]]


def step_env(env: MoonLandingEnv, action: np.ndarray) -> Step:
    """`MoonLandingEnv.step` with its (legacy, untyped) signature pinned."""
    out: Step = env.step(action)  # type: ignore[no-untyped-call]
    return out


def fly_episode(model: Any, env: MoonLandingEnv, site: float) -> dict[str, Any]:
    obs, info = env.reset(options={"site_angle": float(site)})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = step_env(env, action)
        done = terminated or truncated
    return dict(info)


def summarize(rows: list[dict[str, Any]], prefix_return: float) -> Metrics:
    native = float(np.mean([r["native_return"] for r in rows]))
    return {
        "on_site": float(np.mean([r["on_site"] for r in rows])),
        "soft": float(np.mean([r["soft"] for r in rows])),
        "crash": float(np.mean([r["termination"] == "crashed" for r in rows])),
        "timeout": float(
            np.mean([r["termination"] in ("window_limit", "time_limit") for r in rows])
        ),
        "miss_km_med": float(np.median([r["miss_km"] for r in rows])),
        "rel_med": float(np.median([r["moon_rel_speed"] for r in rows])),
        "dv_mean": float(np.mean([r["dv_spent"] for r in rows])),
        "native": native,
        "full_mission": native + prefix_return,
    }


def evaluate(model: Any, env: MoonLandingEnv, episodes: int, verbose: bool = True) -> Metrics:
    """Deterministic rollouts from parking-orbit entry over an even site grid."""
    sites = np.linspace(-math.pi, math.pi, episodes, endpoint=False)
    rows = [fly_episode(model, env, float(s)) for s in sites]
    if verbose:
        for site, r in zip(sites, rows, strict=True):
            print(
                f"  site {math.degrees(site):+7.1f}: {r['termination']:<16} "
                f"rel {r['moon_rel_speed']:6.1f} m/s  miss {r['miss_km']:6.1f} km  "
                f"dv {r['dv_spent']:5.0f}  native {r['native_return']:6.0f}"
            )
    out = summarize(rows, env.bank.prefix_return)
    if verbose:
        print(
            f"  -> on-site {out['on_site']:.0%}  soft {out['soft']:.0%}  "
            f"crash {out['crash']:.0%}  median miss {out['miss_km_med']:.1f} km  "
            f"mean full-mission return {out['full_mission']:.0f}"
        )
    return out


# --- callbacks --------------------------------------------------------------------------

PPO_LOG_KEYS = (
    "train/approx_kl",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance",
    "train/std",
    "train/bound_loss",
    "train/bc_loss",
)


def episode_stats(info: dict[str, Any]) -> dict[str, float]:
    return {
        "on_site": float(info["on_site"]),
        "soft": float(info["soft"]),
        "crash": float(info["termination"] == "crashed"),
        "timeout": float(info["termination"] in ("window_limit", "time_limit")),
        "escape": float(info["termination"] == "escaped"),
        "miss_km": float(info["miss_km"]),
        "rel": float(info["moon_rel_speed"]),
        "ret": float(info["episode"]["r"]),
        "len": float(info["episode"]["l"]),
        "native": float(info["native_return"]),
        "dv": float(info["dv_spent"]),
        "ttl": float(info["start_ttl"]),
    }


class Curriculum(BaseCallback):
    """Raise the start-state level when the policy succeeds often enough at
    the current one, and log training statistics to MLflow.

    The gate reads a deterministic probe (the mean policy) rather than the
    stochastic rollouts, which run 10-15 points below it, and only once the
    noisy rate has cleared half the threshold, since the probe costs about a
    rollout of env steps.
    """

    def __init__(self, cfg: TrainConfig, probe_env: MoonLandingEnv | None, window: int = 200):
        super().__init__()
        self.cfg = cfg
        self.level = cfg.level0
        self.probe_env = probe_env
        self.probe_rate = float("nan")
        self.win = window
        self.recent: list[dict[str, float]] = []
        self.at_level = 0
        self._rollouts = 0
        self._t0 = time.time()
        self.level_history: list[tuple[int, float]] = []

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self.training_env.env_method("set_level", self.level)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.recent.append(episode_stats(info))
                self.at_level += 1
        if len(self.recent) > self.win:
            del self.recent[: -self.win]
        return True

    def _probe(self) -> float:
        assert self.probe_env is not None
        self.probe_env.set_level(self.level)
        hits = 0
        for _ in range(self.cfg.probe_episodes):
            obs, _ = self.probe_env.reset()
            done = False
            info: dict[str, Any] = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = step_env(self.probe_env, action)
                done = terminated or truncated
            hits += int(info["on_site"])
        return hits / self.cfg.probe_episodes

    def _gate_rate(self) -> float:
        """The success rate the advance decision reads this rollout."""
        n = min(len(self.recent), self.at_level) or 1
        noisy = float(np.mean([r["on_site"] for r in self.recent[-n:]]))
        if self.probe_env is None:
            return noisy
        if (
            self.at_level >= self.cfg.min_episodes
            and noisy >= 0.5 * self.cfg.threshold
            and self.level < self.cfg.max_level
            and self._rollouts % self.cfg.probe_every == 0
        ):
            self.probe_rate = self._probe()
            return self.probe_rate
        return -1.0  # no fresh probe: never advance on the noisy rate

    def _advance(self, rate: float) -> None:
        self.probe_rate = float("nan")
        self.level = min(self.cfg.max_level, self.level + self.cfg.level_step)
        self.training_env.env_method("set_level", self.level)
        self.at_level = 0
        self.level_history.append((self.num_timesteps, self.level))
        print(f"  [{self.num_timesteps:>11,}] on-site {rate:.0%} -> level {self.level:.2f}")

    def _log(self) -> None:
        m: Metrics = {
            f"train/{k}": float(np.mean([r[k] for r in self.recent])) for k in self.recent[0]
        }
        m["train/level"] = self.level
        m["train/episodes_at_level"] = float(self.at_level)
        if self.probe_rate == self.probe_rate:
            m["train/probe_on_site"] = self.probe_rate
        for k in PPO_LOG_KEYS:
            v = self.logger.name_to_value.get(k)
            if v is not None:
                m[k.replace("train/", "ppo/")] = float(v)
        m["time/fps"] = self.num_timesteps / max(1e-9, time.time() - self._t0)
        mlflow.log_metrics(m, step=self.num_timesteps)

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if not self.recent:
            return
        rate = self._gate_rate()
        if (
            self.at_level >= self.cfg.min_episodes
            and rate >= self.cfg.threshold
            and self.level < self.cfg.max_level
        ):
            self._advance(rate)
        self._log()


class PeriodicEval(BaseCallback):
    """Every `eval_every` steps: a deterministic evaluation from parking-orbit
    entry, logged to MLflow; `last` and `best` checkpoints saved."""

    def __init__(self, cfg: TrainConfig, out_dir: str):
        super().__init__()
        self.every = cfg.eval_every
        self.episodes = cfg.eval_episodes
        self.out_dir = out_dir
        self._next = cfg.eval_every
        self.env = nominal_env(cfg)
        self.best = -1.0
        self.last: Metrics = {}

    def _save(self, tag: str) -> None:
        self.model.save(os.path.join(self.out_dir, tag))
        vec = self.model.get_vec_normalize_env()
        if vec is not None:
            vec.save(os.path.join(self.out_dir, f"{tag}_vecnorm.pkl"))

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        self.last = evaluate(self.model, self.env, self.episodes, verbose=False)
        mlflow.log_metrics({f"eval/{k}": v for k, v in self.last.items()}, step=self.num_timesteps)
        print(
            f"  [eval @ {self.num_timesteps:>11,}] on-site {self.last['on_site']:.0%} "
            f"soft {self.last['soft']:.0%} crash {self.last['crash']:.0%} "
            f"miss {self.last['miss_km_med']:.1f} km  native {self.last['native']:.0f}"
        )
        self._save("last")
        score = self.last["on_site"] + 0.1 * self.last["soft"]
        if score > self.best:
            self.best = score
            self._save("best")
        return True


# --- behavior cloning ------------------------------------------------------------------


def bc_dataset(bank: StateBank, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """(obs, action) pairs for every bank state, in the task's spaces."""
    env = make_env(cfg, seed=0, level=1.0, state_noise=0.0)()
    obs, act = [], []
    for e in bank.entries:
        restore(env.env, e.snap)
        env.env.set_site_angle(e.site)
        env.start_ttl = e.ttl
        env._t = 0
        env._cap = min(cfg.window, e.ttl + cfg.window_slack)
        obs.append(env._obs())
        act.append(env.expert_action(e))
    return np.stack(obs), np.stack(act)


def bc_pretrain(
    model: PPO, obs: np.ndarray, act: np.ndarray, epochs: int, log_std: float, batch: int = 512
) -> None:
    """Regress the policy mean onto the expert's from-orbit actions."""
    policy = model.policy
    policy.set_training_mode(True)
    obs_t = th.as_tensor(obs, device=policy.device)
    act_t = th.as_tensor(np.clip(act, -0.999, 0.999), device=policy.device)
    opt = th.optim.Adam(policy.parameters(), lr=3e-4)
    n = len(obs_t)
    for epoch in range(epochs):
        perm = th.randperm(n, device=policy.device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            dist = policy.get_distribution(obs_t[idx])
            loss = F.mse_loss(policy_mean(dist), act_t[idx])
            opt.zero_grad()
            backward(loss)
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"      bc epoch {epoch + 1}/{epochs}: mse {tot / n:.5f}")
    with th.no_grad():
        policy.log_std.data.fill_(log_std)
    policy.set_training_mode(False)


# --- a run -------------------------------------------------------------------------------


def load_model(path: str) -> PPO:
    """A saved policy for evaluation (observation normalisation is off in the
    recipe, so the saved reward statistics are not needed to fly it)."""
    return PPO.load(path, device="cpu")


class TrainingRun:
    """One logged training run: build the envs and the model, train with the
    curriculum, evaluate, and close the experiment-log entry."""

    def __init__(self, cfg: TrainConfig, explog: Any):
        self.cfg = cfg
        self.explog = explog
        self.name = cfg.name or time.strftime("ppo_%Y%m%d_%H%M%S")
        self.out_dir = os.path.join("models", self.name)
        self.bank: StateBank = load_bank()

    # -- building -------------------------------------------------------------------------

    def _vec_env(self) -> VecNormalize:
        cfg = self.cfg
        vec = VecMonitor(
            SubprocVecEnv([make_env(cfg, seed=cfg.seed * 1000 + i) for i in range(cfg.envs)])
        )
        return VecNormalize(vec, norm_obs=cfg.norm_obs, norm_reward=True, gamma=cfg.gamma)

    def _model(self, vec: VecNormalize) -> tuple[PPO, VecNormalize]:
        cfg = self.cfg
        algo: type[PPO] = PPO
        if cfg.bound_coef > 0.0:
            algo = BoundedPPO
            BoundedPPO.bound_coef = cfg.bound_coef
            BoundedPPO.std_floor = max(1e-3, cfg.std_floor)
        if cfg.resume:
            base = cfg.resume[:-4] if cfg.resume.endswith(".zip") else cfg.resume
            stats = base + "_vecnorm.pkl"
            if os.path.exists(stats):
                vec = VecNormalize.load(stats, vec.venv)
                vec.training = True
                print(f"      restored reward-normalisation stats from {stats}")
            model = algo.load(cfg.resume, env=vec, device="cpu", learning_rate=cfg.lr)
            if cfg.bound_coef > 0.0:
                with th.no_grad():
                    model.policy.log_std.data.clamp_(min=math.log(BoundedPPO.std_floor))
            return model, vec
        model = algo(
            "MlpPolicy",
            vec,
            learning_rate=cfg.lr,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=dict(
                net_arch=dict(pi=list(cfg.net), vf=list(cfg.net)), log_std_init=cfg.log_std_init
            ),
            seed=cfg.seed,
            device="cpu",
            verbose=0,
            use_sde=cfg.sde,
            sde_sample_freq=cfg.sde_freq if cfg.sde else -1,
        )
        return model, vec

    def _warm_start(self, model: PPO) -> None:
        cfg = self.cfg
        if cfg.bc_epochs <= 0 and cfg.bc_coef <= 0.0:
            return
        obs, act = bc_dataset(self.bank, cfg)
        if cfg.bc_epochs > 0 and not cfg.resume:
            print(f"      behavior cloning on {len(obs)} expert states, {cfg.bc_epochs} epochs")
            bc_pretrain(model, obs, act, cfg.bc_epochs, log_std=cfg.log_std_init)
        if cfg.bc_coef > 0.0 and isinstance(model, BoundedPPO):
            model.set_bc_anchor(obs, act, cfg.bc_coef)
            print(f"      BC anchor on {len(obs)} expert pairs, coef {cfg.bc_coef}")

    # -- running --------------------------------------------------------------------------

    def run(self) -> Metrics:
        cfg = self.cfg
        print(
            f"[1/3] expert state bank: {len(self.bank.entries)} states, descent "
            f"{self.bank.descent_len} steps, max ttl {self.bank.max_ttl}"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        params = {
            **cfg.flat(),
            "bank_states": len(self.bank.entries),
            "descent_len": self.bank.descent_len,
            "max_ttl": self.bank.max_ttl,
        }
        self.explog.start(
            self.name, params, cfg.hypothesis, tags={"algo": "PPO", "task": "orbit->site"}
        )
        print("[2/3] building vec env + PPO")
        model, vec = self._model(self._vec_env())
        self._warm_start(model)
        probe_env = make_env(cfg, seed=777)() if cfg.probe_episodes > 0 else None
        curriculum = Curriculum(cfg, probe_env)
        periodic = PeriodicEval(cfg, self.out_dir)
        print(f"[3/3] training {cfg.steps:,} steps -> {self.out_dir}")
        t0 = time.time()
        status, outcome = "FINISHED", ""
        try:
            model.learn(total_timesteps=cfg.steps, callback=[curriculum, periodic])
        except KeyboardInterrupt:
            status, outcome = "KILLED", "interrupted"
        finally:
            final = self._finish(model, vec, curriculum, periodic, status, outcome, t0)
        return final

    def _finish(
        self,
        model: PPO,
        vec: VecNormalize,
        curriculum: Curriculum,
        periodic: PeriodicEval,
        status: str,
        outcome: str,
        t0: float,
    ) -> Metrics:
        model.save(os.path.join(self.out_dir, "final"))
        vec.save(os.path.join(self.out_dir, "final_vecnorm.pkl"))
        print("final evaluation (parking orbit, 32 sites):")
        final = evaluate(model, periodic.env, 32)
        mlflow.log_metrics({f"final/{k}": v for k, v in final.items()}, step=model.num_timesteps)
        mlflow.log_artifacts(self.out_dir)
        mins = (time.time() - t0) / 60
        outcome = outcome or (
            f"level reached {curriculum.level:.2f}; final on-site {final['on_site']:.0%}, "
            f"soft {final['soft']:.0%}, crash {final['crash']:.0%}, "
            f"median miss {final['miss_km_med']:.1f} km, "
            f"{model.num_timesteps:,} steps in {mins:.0f} min"
        )
        history = ", ".join(f"{s // 1000}k:{lvl:.2f}" for s, lvl in curriculum.level_history[-12:])
        self.explog.finish(
            outcome,
            f"(fill in with `python scripts/explog.py diagnosis <run> ...`) level history: {history}",
            status,
            summary_metrics={
                "final/level": curriculum.level,
                **{f"final/{k}": v for k, v in final.items()},
            },
        )
        print(f"saved {self.out_dir}/final.zip; {outcome}")
        return final


def evaluate_checkpoint(cfg: TrainConfig, path: str, episodes: int, watch: bool = False) -> Metrics:
    """Fly a saved policy from parking-orbit entry over a site grid."""
    env = nominal_env(cfg, render_mode="human" if watch else None)
    return evaluate(load_model(path), env, episodes)
