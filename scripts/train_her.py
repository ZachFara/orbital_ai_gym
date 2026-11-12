"""Goal-conditioned TD3 + HER for the lunar landing site (ideas #1 and #3, mixed).

`train_guided.py` learns ONE site with on-policy PPO: every off-site landing is
a near-miss with weak shaping, and PPO throws each rollout away after one pass.
This script attacks site targeting the two sample-efficient ways that actually
fit a deterministic, known-dynamics env with a scripted expert:

  #1  Off-policy learning with a replay buffer, pre-seeded with expert landings
      and behavior-cloned before the first gradient step. Every transition is
      reused many times instead of once, and the critic sees real landings
      immediately.

  #3  Goal-conditioned site + Hindsight Experience Replay. The site is fed to
      the policy as a goal and randomized per episode; HER relabels every
      landing as a *success for the site it actually hit*. A trajectory that
      misses the commanded site is still a perfect demonstration of "how to
      land at THAT spot", so every episode yields a positive example instead
      of a near-miss. The site changes only the reward and the observation,
      never the dynamics, so randomizing it leaves the cached expert-action
      approach prefix exactly valid (unlike randomizing the moon's distance).

The mix: the replay buffer IS a HerReplayBuffer, seeded with expert landings,
so the first gradient steps already see hindsight-relabeled successes.

Three measurements shaped this design (see README):

  * The touchdown is brittle. The autopilot rides its suicide-burn profile to
    the ground at exactly 8 m/s against a 25 m/s limit; sigma=0.005 action
    noise soft-lands only ~50% of the time and sigma=0.01 never does. SAC's
    auto-entropy exploration targets far more noise than that, so this uses
    TD3, whose exploration sigma is set explicitly (default 0.005).
  * The autopilot cannot steer. It touches down at the same body-frame bearing
    (~127 deg) no matter where its site is set; aiming it anywhere else only
    engages a lateral term that wrecks the landing. So it is used purely as a
    soft-landing demonstrator and the goal is decoupled from its aim --
    targeting is entirely the policy's job, as it always was.
  * The reachable goal arc is measurable. Perturbing the approach (noise early,
    none during terminal descent) still lands softly but moves the touchdown
    bearing, so the demos' achieved bearings define the goal distribution
    empirically instead of by guesswork.

Reward is sparse and gated: 0 only on a soft touchdown whose sub-point is
within the site radius, -1 otherwise. The `landed_soft` gate in info is what
keeps HER honest -- relabeling a 300 m/s crash as "success for where it
crashed" would teach the policy to slam into the ground, so a crash stays -1
for every relabeled goal. That gate is why `copy_info_dict=True` is mandatory.

Training stays in the landing window (a cached expert prefix flies the
approach), which is where site targeting lives. Ascent/orbit/transfer remain
the autopilot's job, as in train_guided.

    pip install -e ".[train]"
    python scripts/train_her.py --steps 300000 --out models/moon_her
    python scripts/train_her.py --eval 20 --watch models/moon_her.zip
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from typing import Any, ClassVar

import numpy as np

try:
    import torch as th
    from stable_baselines3 import TD3, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import polyak_update
    from stable_baselines3.common.vec_env import DummyVecEnv
    from torch.nn import functional as F
except ImportError:
    raise SystemExit("stable-baselines3 not installed; run: pip install -e '.[train]'") from None

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autopilot_demo import MoonAutopilot

from orbital_gym.missions.to_moon import ToMoonEnv

GOAL_DIM = 2  # the site as a unit (cos, sin) bearing in the moon body frame
SOFT_TERMS = ("landed_moon", "landed_moon_site")


# --- goal geometry (shared by the env, the demo collector, and eval) ---------


def body_bearing(u: ToMoonEnv) -> float:
    """The rocket's sub-point as an angle in the moon's BODY frame.

    Subtracting the moon's orbital angle makes this drift-free: it is directly
    comparable to the site (a constant body-frame angle) wherever the moon is
    on its orbit. At touchdown it is the landing longitude.
    """
    mx, my = u._moon_world()
    b = math.atan2(u._ry - my, u._rx - mx) - u._moon_angle()
    return math.atan2(math.sin(b), math.cos(b))


def achieved_bearing(u: ToMoonEnv) -> float:
    """Where this trajectory is currently headed, in the moon's body frame.

    The sub-point alone is a bad achieved-goal: it only equals the landing site
    on the very last step, so HER could relabel just one transition per episode
    into a success (measured: 0.15% of samples -- far too thin to learn from).
    The env's conic impact predictor answers "where would I touch down if I
    stopped thrusting" at EVERY step, so "am I aimed at the site" is a question
    with an answer the whole way down, and relabeling makes whole stretches of
    a trajectory informative. Falls back to the sub-point when no impact is
    predicted (high coast, or already on the surface).
    """
    impacting, miss = u._moon_impact()
    if impacting:
        # miss is signed from the site; adding it back recovers an absolute
        # body-frame bearing that does not depend on where the site is
        return math.atan2(math.sin(u._site_angle + miss), math.cos(u._site_angle + miss))
    return body_bearing(u)


def _achieved_cossin(u: ToMoonEnv) -> np.ndarray:
    b = achieved_bearing(u)
    return np.array([math.cos(b), math.sin(b)], dtype=np.float32)


def _desired_cossin(u: ToMoonEnv) -> np.ndarray:
    return np.array([math.cos(u._site_angle), math.sin(u._site_angle)], dtype=np.float32)


def goal_obs(u: ToMoonEnv) -> dict:
    return {
        "observation": u._obs().astype(np.float32),
        "achieved_goal": _achieved_cossin(u),
        "desired_goal": _desired_cossin(u),
    }


def landed_soft(info: dict) -> bool:
    return info.get("termination") in SOFT_TERMS


# Terminal magnitudes, set from --reward-scale. They sit against a dense term
# bounded by 1, and the critic has to fit both through bootstrapping: at the
# original +-50 that is a 50:1 ratio, which is a classic way to destabilize a
# TD critic. Kept module-level (not threaded through) so main() can retune
# them once and every caller -- env, HER relabeling, demo replay -- agrees.
SUCCESS_BONUS = 5.0
FAILURE_PENALTY = 5.0
CRASH_SPEED_REF = 500.0  # impact speed at which the crash penalty saturates
# A marginal crash must still cost more than a soft landing earns, or "crash
# slowly near the site" beats "land softly further away" -- measured, a
# 65 m/s crash scored +0.96 against -2.14 for a soft touchdown 19 km out.
CRASH_PENALTY_FLOOR = 0.3
SOFT_BONUS_FRAC = 0.2  # credit for a soft touchdown anywhere, vs on-site
DENSE_SITE_RADII = 3.0  # shaping denominator, in site radii (~the goal arc)
DENSE_STEP_CLIP = 0.2  # no single step may dominate the outcome
# Residual-mode goal arc, rad. ~8.6 deg = ~24 km of surface, three times the
# 8 km site radius: a do-nothing policy lands on the commanded site only about
# a third of the time, so there is real targeting left to learn.
RESIDUAL_ARC = 0.15

# Dense soft-descent shaping. The touchdown is knife-edge (~1 m/s vs 25) and the
# terminal soft/crash signal is only felt at the very end, so the policy has to
# STUMBLE onto a soft descent -- which exploration tight enough to survive can't
# do. This makes "stay landable" a dense gradient: a potential on how far the
# moon-relative speed exceeds what can still be braked to a stop by the surface
# (v_prof = sqrt(2*a_net*altitude)). Zero on a landable descent, sharply
# negative on a dive. Potential-based (reward is the change), so per Ng et al.
# it adds gradient WITHOUT changing the optimal policy and cannot be farmed.
DESCENT_SHAPE_W = 2.0  # weight; set from --descent-shape-w (0 disables)
DESCENT_V_SCALE = 300.0  # m/s of excess speed that scores one potential unit
DESCENT_SAFE = 0.95  # aim to stay under this fraction of the stop profile


def descent_potential(u) -> float:
    """Landability potential in the moon frame: 0 when the moon-relative speed
    is within what can still be braked to a stop by the surface, negative (and
    growing) as the descent dives past it. Goal-independent, so it shapes the
    descent without touching the site-targeting reward."""
    mvx, mvy = u._moon_velocity()
    v = math.hypot(u._vx - mvx, u._vy - mvy)
    d = u._moon_distance()
    if d > u.moon_soi:
        return 0.0  # only shape once near the moon
    malt = max(1.0, d - u.moon.radius)
    a_max = u.rocket.stages[u._stage].thrust / u._mass
    g_m = u.moon.mu / (d * d)
    a_net = max(0.5, 0.85 * a_max - g_m)
    v_prof = math.sqrt(2.0 * a_net * malt)
    excess = max(0.0, v - DESCENT_SAFE * v_prof)
    return -min(3.0, excess / DESCENT_V_SCALE)


def goal_arc(achieved: np.ndarray, desired: np.ndarray, moon_radius: float) -> np.ndarray:
    """Surface arc, in meters, between two body-frame bearings.

    Both are unit (cos, sin), so their dot is the cosine of the angle between
    them and R*arccos(dot) is the arc along the surface.
    """
    achieved = np.asarray(achieved, dtype=np.float64).reshape(-1, GOAL_DIM)
    desired = np.asarray(desired, dtype=np.float64).reshape(-1, GOAL_DIM)
    dot = np.clip(np.sum(achieved * desired, axis=1), -1.0, 1.0)
    return moon_radius * np.arccos(dot)


def goal_success(achieved, desired, infos, moon_radius, site_radius) -> np.ndarray:
    soft = np.array([bool(i.get("landed_soft", False)) for i in infos])
    return soft & (goal_arc(achieved, desired, moon_radius) <= site_radius)


def goal_reward(
    achieved: np.ndarray, desired: np.ndarray, infos: list, moon_radius: float, site_radius: float
) -> np.ndarray:
    """Vectorized HER reward, dense in the goal but potential-based.

    Per step: how much CLOSER the predicted touchdown moved to the site, in
    site-radii. It is a function of the goal at every step, so relabeling a
    trajectory produces a trajectory of informative rewards -- a purely
    terminal criterion gave HER one usable transition in ~480 (0.15%), which
    is not learnable.

    The difference form matters. As a per-step level (-arc/(pi*R)) it summed
    over ~592 steps to about -12 at a 10 km miss, dwarfing a -5 crash, so
    wrecking the landing to shave the miss was a winning trade and the policy
    took it (soft landings went 12/12 -> 0/12 while the miss improved). In
    difference form it telescopes to a bounded total, leaving the terminal
    outcome dominant -- the same potential-based shaping the missions
    themselves use so progress rewards cannot be farmed.

    Terminal: +SUCCESS_BONUS for a soft touchdown inside the site radius,
    -FAILURE_PENALTY for ending the episode ANY other way. The penalty must
    cover every non-landing terminal, not just "crashed": when only crashes
    were penalized the policy learned to escape the moon entirely (measured:
    3-4 km/s outbound, 660-980 km off site, getting worse with training),
    because fleeing ended the episode for free while attempting a landing
    risked -50. Any unpenalized exit is strictly better than a hard problem.

    The failure term is also what keeps HER honest: relabeling a 300 m/s
    impact as "the goal was where it hit" would otherwise reward slamming into
    the ground, so a crash stays heavily negative for every relabeled goal.
    """
    arc = goal_arc(achieved, desired, moon_radius)
    # the previous step's aim point, carried in info so relabeled goals get a
    # consistent before/after pair
    prev = np.stack(
        [
            np.asarray(i.get("prev_achieved", a), dtype=np.float64)
            for i, a in zip(infos, np.asarray(achieved).reshape(-1, GOAL_DIM), strict=True)
        ]
    )
    arc_prev = goal_arc(prev, desired, moon_radius)
    reward = np.clip(
        (arc_prev - arc) / (DENSE_SITE_RADII * site_radius), -DENSE_STEP_CLIP, DENSE_STEP_CLIP
    )
    soft = np.array([bool(i.get("landed_soft", False)) for i in infos])
    # a terminal step is any step the env labeled; non-terminal steps have None
    ended = np.array([bool(i.get("termination")) for i in infos])
    crashed = np.array([i.get("termination") == "crashed" for i in infos])
    speed = np.array([float(i.get("moon_rel_speed", CRASH_SPEED_REF)) for i in infos])

    # Grade the crash penalty by impact speed. A flat penalty makes a 430 m/s
    # arrival and a 1560 m/s arrival score identically, so nothing pushes the
    # policy toward touching down gently -- measured, impact speed drifted
    # UPWARD during training under a flat penalty. Scaling it turns "land
    # softer" into a gradient the critic can actually follow. Non-impact
    # failures (escape, running out the window) take the full penalty.
    scale = np.where(crashed, np.clip(speed / CRASH_SPEED_REF, CRASH_PENALTY_FLOOR, 1.0), 1.0)
    on_site = soft & (arc <= site_radius)
    reward = reward + SUCCESS_BONUS * on_site
    # a soft touchdown anywhere is still a landing, and must beat any crash
    reward = reward + SUCCESS_BONUS * SOFT_BONUS_FRAC * (soft & ~on_site)
    reward = reward - FAILURE_PENALTY * scale * (ended & ~soft)
    # dense soft-descent shaping, precomputed per step and stored in info so it
    # is goal-independent (HER relabeling leaves it untouched, as it should --
    # how softly you descend does not depend on which site you aimed at)
    reward = reward + np.array([float(i.get("descent_shaping", 0.0)) for i in infos])
    return reward.astype(np.float32)


# --- TD3+BC: keep the actor anchored to the demonstrations -------------------


class TD3BC(TD3):
    """TD3 with the Fujimoto TD3+BC actor loss.

    Plain TD3 destroys a behavior-cloned policy almost immediately here: the
    critic starts random, and the actor is trained to maximize those garbage
    Q-values. Measured, it collapsed to a constant "never brake" policy within
    3k steps (every episode impacting at exactly 1705.7 m/s) despite starting
    from a BC clone with 0.010 MSE.

    The fix is one extra term: regress the actor toward the actions actually
    in the replay batch while it chases Q, with the Q term normalized by its
    own scale so the balance doesn't drift as returns grow.

        actor_loss = -lambda * Q1(s, pi(s)).mean() + MSE(pi(s), a_batch)
        lambda     = bc_alpha / mean|Q1|

    Lower bc_alpha holds tighter to the demos. Because the buffer is seeded
    with expert landings and fed by a policy that starts as their clone, the
    BC term anchors to competent behavior rather than to noise.
    """

    bc_alpha = 2.5  # class attribute, not a ctor kwarg, so .load() round-trips
    # lambda = bc_alpha / mean|Q| explodes while the critic is untrained and
    # mean|Q| ~ 0, which swamps the BC anchor exactly when it is needed most
    # (measured: collapse by 3k steps even WITH the BC term). The cap exists
    # only to survive that startup window -- set it too low and it never
    # releases: at mean|Q| = 0.63 a cap of 1.0 held lambda at 1.0 instead of
    # 3.97 and throttled the RL term permanently, leaving the residual pinned
    # near zero while landings stayed perfect and nothing was learned.
    bc_lambda_max = 10.0
    # Weight on the BC term during training. Direct mode needs it (the anchor
    # is the expert). Residual mode must NOT: there the expert is a ZERO
    # residual, so the same term is a straight penalty on acting at all --
    # measured, the policy used 4% of its authority and never beat the
    # do-nothing baseline. BC still sets the starting point via bc_pretrain;
    # what bounds the policy afterwards is residual_scale, not this.
    bc_weight = 1.0
    demo_obs: dict | None = None
    demo_act = None

    def set_demos(self, obs_list: list[dict], act: list[np.ndarray]) -> None:
        """Pin the BC anchor to the expert demonstrations.

        Fujimoto's TD3+BC regresses toward the actions in the sampled replay
        batch, which is correct OFFLINE where the buffer is a fixed good
        dataset. Online it is a feedback loop: the buffer fills with the
        current policy's own transitions, so a degrading actor is anchored to
        its own degradation. Measured, the policy collapsed the same way under
        every bc_alpha because the anchor was drifting with it. Regressing
        against the demonstrations instead keeps the reference fixed.
        """
        dev = self.device
        self.demo_obs = {
            k: th.as_tensor(np.stack([o[k] for o in obs_list]), device=dev) for k in obs_list[0]
        }
        self.demo_act = th.as_tensor(np.clip(np.stack(act), -0.999, 0.999), device=dev)

    def warmup_critic(self, gradient_steps: int, batch_size: int = 512) -> None:
        """Fit the critic on the seeded buffer before the actor may move.

        The actor is a behavior-cloned soft-lander at this point; letting it
        chase an untrained critic throws that away. This gives Q a real scale
        first, using only demonstration data.
        """
        self.policy.set_training_mode(True)
        for step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                next_q = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q
            current_q = self.critic(replay_data.observations, replay_data.actions)
            loss = sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            if (step + 1) % max(1, gradient_steps // 5) == 0:
                with th.no_grad():
                    scale = (
                        self.critic.q1_forward(replay_data.observations, replay_data.actions)
                        .abs()
                        .mean()
                    )
                print(
                    f"  critic warmup {step + 1}/{gradient_steps}: "
                    f"loss {loss.item():.4f}  mean|Q| {scale.item():.2f}"
                )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, bc_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(
                    -1, 1
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                pi = self.actor(replay_data.observations)
                q_pi = self.critic.q1_forward(replay_data.observations, pi)
                lmbda = th.clamp(self.bc_alpha / q_pi.abs().mean().detach(), max=self.bc_lambda_max)
                if self.bc_weight <= 0.0:
                    bc_loss = th.zeros((), device=self.device)
                elif self.demo_act is not None:
                    idx = th.randint(0, len(self.demo_act), (batch_size,), device=self.device)
                    bc_loss = F.mse_loss(
                        self.actor({k: v[idx] for k, v in self.demo_obs.items()}),
                        self.demo_act[idx],
                    )
                else:
                    bc_loss = F.mse_loss(pi, replay_data.actions)
                actor_loss = -lmbda * q_pi.mean() + self.bc_weight * bc_loss
                actor_losses.append(actor_loss.item())
                bc_losses.append(bc_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(
                    self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0
                )
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


# --- the goal-conditioned landing-window environment -------------------------


class GoalMoonEnv(gym.Env):
    """ToMoon as a HER GoalEnv, restricted to the landing window.

    reset() restores a snapshot of the env at the expert handoff (the approach
    is flown once, cached, and reused -- dynamics don't depend on the site, so
    one snapshot serves every goal) and draws a new random site. step() returns
    the sparse gated reward. compute_reward() is the same function HER calls to
    score relabeled goals.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    SNAP_SKIP = ("_renderer",)

    def __init__(
        self,
        actions: list[np.ndarray],
        handoff: float,
        site_center: float,
        site_arc: float,
        max_steps: int,
        site_radius: float | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        n_pre: int | None = None,
        env_max_steps: int | None = None,
        hover_control: bool = False,
    ):
        super().__init__()
        kw = dict(render_mode=render_mode, hover_control=hover_control)
        if env_max_steps is not None:
            kw["max_episode_steps"] = env_max_steps
        self.env = ToMoonEnv(**kw)
        u = self.env
        self._actions = actions
        # orbit-first hands off at a fixed step (parking-orbit entry); the
        # window modes hand off at a fraction of the expert episode
        self._n_pre = n_pre if n_pre is not None else int(handoff * len(actions))
        self.site_center = site_center
        self.site_arc = site_arc
        self.max_steps = max_steps
        self.moon_radius = u.moon.radius
        self.site_radius = u.moon_site_radius if site_radius is None else site_radius
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._snap = None
        self._t = 0
        self._desired = _desired_cossin(u)

        self.action_space = u.action_space
        self.observation_space = spaces.Dict(
            {
                "observation": u.observation_space,
                "achieved_goal": spaces.Box(-1.0, 1.0, (GOAL_DIM,), np.float32),
                "desired_goal": spaces.Box(-1.0, 1.0, (GOAL_DIM,), np.float32),
            }
        )

    def set_site_arc(self, center: float, arc: float) -> None:
        self.site_center, self.site_arc = float(center), float(arc)

    def _ensure_snapshot(self):
        if self._snap is not None:
            return
        u = self.env
        u.reset()
        for i in range(self._n_pre):
            _, _, term, trunc, _ = u.step(self._actions[i])
            if term or trunc:  # expert episode ended inside the prefix
                break
        self._snap = {
            k: copy.deepcopy(v)
            for k, v in u.__dict__.items()
            if k.startswith("_") and k not in self.SNAP_SKIP
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._ensure_snapshot()
        u = self.env
        u.__dict__.update(copy.deepcopy(self._snap))
        site = (
            float(options["site_angle"])
            if options and "site_angle" in options
            else self.site_center + self._rng.uniform(-self.site_arc, self.site_arc)
        )
        u.set_site_angle(site)
        u._last_site_d = u._moon_site_distance()
        self._desired = _desired_cossin(u)
        self._t = 0
        self._prev_achieved = _achieved_cossin(u)
        self._prev_phi = descent_potential(u)
        info = dict(u._info())
        info["landed_soft"] = False
        self._last_info = info
        return goal_obs(u), info

    def _env_action(self, action):
        """What actually reaches the env. Identity here; the residual variant
        adds the policy's correction on top of the autopilot."""
        return action

    def step(self, action):
        u = self.env
        prev_achieved = self._prev_achieved
        _, _, terminated, truncated, info = self.env.step(self._env_action(action))
        self._t += 1
        info = dict(info)
        if self._t >= self.max_steps and not terminated:
            # the window cap is the wrapper's, not the env's, so the env never
            # sets a terminal label for it -- name it rather than leave None
            truncated = True
            info["termination"] = info.get("termination") or "window_limit"
        info["landed_soft"] = landed_soft(info)
        info["prev_achieved"] = prev_achieved
        phi = descent_potential(u)
        info["descent_shaping"] = DESCENT_SHAPE_W * (phi - self._prev_phi)
        self._prev_phi = phi
        gobs = goal_obs(u)
        self._prev_achieved = gobs["achieved_goal"]
        reward = float(
            goal_reward(
                gobs["achieved_goal"], self._desired, [info], self.moon_radius, self.site_radius
            )[0]
        )
        info["is_success"] = bool(
            goal_success(
                gobs["achieved_goal"], self._desired, [info], self.moon_radius, self.site_radius
            )[0]
        )
        self._last_info = info
        return gobs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this with batched goals and a list of stored infos.
        infos = info if isinstance(info, (list, np.ndarray)) else [info]
        return goal_reward(achieved_goal, desired_goal, infos, self.moon_radius, self.site_radius)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class ResidualGoalMoonEnv(GoalMoonEnv):
    """The policy corrects the autopilot instead of replacing it.

        env_action = clip(pilot(info) + residual_scale * policy(obs), -1, 1)

    This exists because cloning the expert is what kept failing. The landing
    rides the suicide-burn profile to the ground at 8 m/s against a 25 m/s
    limit, and a 0.010-MSE clone still arrived at ~450 m/s; TD3 then had to
    cross 450 -> 25 m/s by exploration from a buffer containing no successes
    of its own, which it never did in 150k steps.

    A ZERO residual is exactly the autopilot, so the policy starts already
    soft-landing: exploration begins from success, every seeded episode is a
    real landing, and the correction only has to move the touchdown, not
    rediscover how to touch down. The residual is bounded because the profile
    is delicate -- measured, 0.005 of raw action noise already shifts the
    touchdown ~15 degrees, so a small correction has plenty of authority.

    The autopilot's mode machine is stateful, so a pilot constructed fresh at
    the handoff would think it was still on the launch pad. Its state is
    snapshotted alongside the env's and restored on every reset.
    """

    def __init__(self, *args, residual_scale: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_scale = residual_scale
        self.pilot = MoonAutopilot(self.env)
        self._pilot_snap = None

    def _ensure_snapshot(self):
        if self._snap is not None:
            return
        u = self.env
        _, info = u.reset()
        self.pilot = MoonAutopilot(u)
        # fly the approach with a LIVE pilot so its mode machine is warmed up
        for _ in range(self._n_pre):
            _, _, term, trunc, info = u.step(self.pilot(info))
            if term or trunc:
                break
        self._snap = {
            k: copy.deepcopy(v)
            for k, v in u.__dict__.items()
            if k.startswith("_") and k not in self.SNAP_SKIP
        }
        self._pilot_snap = {
            k: copy.deepcopy(v) for k, v in self.pilot.__dict__.items() if k != "env"
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.pilot.__dict__.update({k: copy.deepcopy(v) for k, v in self._pilot_snap.items()})
        self._last_info = dict(self.env._info())
        self._last_info["landed_soft"] = False
        return obs, info

    def _env_action(self, action):
        base = np.asarray(self.pilot(self._last_info), dtype=np.float64)
        res = np.asarray(action, dtype=np.float64).flatten()
        return np.clip(base + self.residual_scale * res, -1.0, 1.0).astype(np.float32)


def collect_residual_demos(env: ResidualGoalMoonEnv, episodes: int, noise: float, seed: int):
    """Fly the residual env with a near-zero residual.

    The autopilot is doing the flying, so essentially every episode is a real
    soft landing -- the thing 150k steps of clone-and-improve never produced.
    Actions stored are the RESIDUALS (what the policy controls), not the
    combined command.
    """
    rng = np.random.default_rng(seed)
    n_act = int(env.action_space.shape[0])
    episodes_out, bc_obs, bc_act, bearings = [], [], [], []
    softs = 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        traj = []
        while not done:
            res = (
                np.zeros(n_act, np.float32)
                if noise <= 0
                else np.clip(rng.normal(0.0, noise, n_act), -1.0, 1.0).astype(np.float32)
            )
            nxt, r, term, trunc, info = env.step(res)
            done = term or trunc
            traj.append((obs, nxt, res, r, done, info))
            bc_obs.append(obs)
            bc_act.append(res)
            obs = nxt
        if landed_soft(info):
            softs += 1
            bearings.append(body_bearing(env.env))
        episodes_out.append(traj)
    return episodes_out, bc_obs, bc_act, bearings, softs


# --- expert flights ----------------------------------------------------------


def fly_expert(
    noise: float = 0.0,
    tail: int = 300,
    seed: int = 0,
    site_angle: float = 0.0,
    n_nominal: int | None = None,
):
    """One autopilot flight. Noise is applied to the executed actions only
    BEFORE the last `tail` steps: the terminal descent rides the suicide-burn
    profile with no margin, so perturbing it just crashes, while perturbing the
    approach moves the touchdown bearing and the pilot still lands softly.

    Returns (exec_actions, clean_actions, info, bearing, soft).
    """
    rng = np.random.default_rng(seed)
    env = ToMoonEnv()
    _, info = env.reset(options={"site_angle": site_angle})
    pilot = MoonAutopilot(env)
    execs, cleans = [], []
    i = 0
    while True:
        clean = np.asarray(pilot(info), dtype=np.float32)
        a = clean
        if noise > 0.0 and (n_nominal is None or i < n_nominal - tail):
            a = np.clip(clean + rng.normal(0.0, noise, clean.shape), -1.0, 1.0).astype(np.float32)
        execs.append(a)
        cleans.append(clean)
        _, _, term, trunc, info = env.step(a)
        i += 1
        if term or trunc:
            break
    b = body_bearing(env)
    env.close()
    return execs, cleans, info, b, landed_soft(info)


def expert_actions() -> list[np.ndarray]:
    """The clean expert episode as an action sequence (deterministic env)."""
    return fly_expert()[0]


def natural_touchdown_bearing() -> float:
    """Body-frame longitude a clean autopilot flight touches down at. Used as
    a provisional goal-arc center before the demos measure the real span."""
    return fly_expert()[3]


# --- orbit-first expert (the whole point of the from-orbit RL setup) ---------

ORBIT_ENV_STEPS = 12_000  # the orbit flight is ~6250 steps; leave room to coast


def orbit_expert(site_angle: float = 0.0, hover_control: bool = False):
    """Fly the orbit-first autopilot: it captures into a low moon orbit, waits,
    deorbits timed for the site, and lands ON it (~1 m/s, any site).

    Returns (actions, n_orbit, info): the action sequence, the step index where
    it enters the parking orbit (the RL handoff point -- RL learns everything
    from there: when to deorbit and how to descend), and the terminal info.
    """
    env = ToMoonEnv(max_episode_steps=ORBIT_ENV_STEPS, hover_control=hover_control)
    _, info = env.reset(options={"site_angle": site_angle})
    pilot = MoonAutopilot(env, orbit=True)
    actions, n_orbit = [], None
    post_capture = ("moon_orbit", "deorbit", "pdescend")
    while True:
        a = np.asarray(pilot(info), dtype=np.float32)
        actions.append(a)
        # capture-complete is the handoff, whether or not it lingers in
        # moon_orbit -- for some sites the deorbit trigger fires the instant
        # the orbit is reached, skipping the coast in a single step
        if n_orbit is None and pilot.mode in post_capture:
            n_orbit = len(actions) - 1
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    env.close()
    if n_orbit is None:
        raise SystemExit("orbit expert never reached a parking orbit")
    return actions, n_orbit, info


def collect_orbit_demos(
    n_pre: int, site_radius: float, max_steps: int, sites, seed: int, hover_control: bool = False
):
    """Fly the orbit expert at each site and keep the from-orbit segment
    (handoff onward) as buffer transitions + BC pairs. The expert lands on the
    site for every site, so every demo is an on-site soft landing -- ideal HER
    seeds, and the whole reachable arc is covered by construction."""
    moon_radius = ToMoonEnv().moon.radius
    all_trans, all_bc_obs, all_bc_act, bearings = [], [], [], []
    softs = 0
    for site in sites:
        acts, _n_orbit, _ = orbit_expert(site, hover_control=hover_control)
        env = ToMoonEnv(max_episode_steps=ORBIT_ENV_STEPS, hover_control=hover_control)
        env.reset(options={"site_angle": site})
        obs = goal_obs(env)
        traj = []
        prev = obs["achieved_goal"]
        prev_phi = descent_potential(env)
        for i, a in enumerate(acts):
            _, _, term, trunc, info = env.step(a)
            info = dict(info)
            info["landed_soft"] = landed_soft(info)
            info["prev_achieved"] = prev
            phi = descent_potential(env)
            info["descent_shaping"] = DESCENT_SHAPE_W * (phi - prev_phi)
            prev_phi = phi
            nxt = goal_obs(env)
            if i >= n_pre:
                r = float(
                    goal_reward(
                        nxt["achieved_goal"], nxt["desired_goal"], [info], moon_radius, site_radius
                    )[0]
                )
                done = term or trunc or (i - n_pre + 1) >= max_steps
                traj.append((obs, nxt, a, r, done, info))
                all_bc_obs.append(obs)
                all_bc_act.append(a)
            obs = nxt
            prev = nxt["achieved_goal"]
            if term or trunc or (i - n_pre + 1) >= max_steps:
                break
        if traj:
            o, no, a, r, _, ii = traj[-1]
            traj[-1] = (o, no, a, r, True, ii)
        if landed_soft(info):
            softs += 1
            bearings.append(body_bearing(env))
        all_trans.append(traj)
        env.close()
    return all_trans, all_bc_obs, all_bc_act, bearings, softs


def replay_window(
    exec_acts,
    clean_acts,
    site: float,
    n_pre: int,
    max_steps: int,
    moon_radius: float,
    site_radius: float,
):
    """Re-fly a recorded action sequence with the site set, collecting the
    landing-window transitions (buffer) and clean-label pairs (BC).

    The env is deterministic and the site never touches dynamics, so this
    reproduces the recorded landing exactly while giving observations that
    reflect the commanded goal.
    """
    env = ToMoonEnv()
    env.reset(options={"site_angle": site})
    obs = goal_obs(env)
    transitions, bc_obs, bc_act = [], [], []
    for i, a in enumerate(exec_acts):
        _, _, term, trunc, info = env.step(a)
        info = dict(info)
        info["landed_soft"] = landed_soft(info)
        nxt = goal_obs(env)
        if i >= n_pre:
            r = float(
                goal_reward(
                    nxt["achieved_goal"], nxt["desired_goal"], [info], moon_radius, site_radius
                )[0]
            )
            done = term or trunc or (i - n_pre + 1) >= max_steps
            transitions.append((obs, nxt, a, r, done, info))
            bc_obs.append(obs)
            bc_act.append(clean_acts[i])
        obs = nxt
        if term or trunc or (i - n_pre + 1) >= max_steps:
            break
    env.close()
    if transitions:  # close the buffer episode
        o, no, a, r, _, i_ = transitions[-1]
        transitions[-1] = (o, no, a, r, True, i_)
    return transitions, bc_obs, bc_act


def derive_arc(
    bearings: list[float], site_arc: float | None, arc_margin: float
) -> tuple[float, float]:
    """The goal arc is the span the demos actually landed in -- wrap-safe, and
    tight rather than generous: commanding goals no trajectory in the buffer
    ever reached would train against targets nothing demonstrates."""
    cm = math.atan2(float(np.mean(np.sin(bearings))), float(np.mean(np.cos(bearings))))
    dev = [math.atan2(math.sin(b - cm), math.cos(b - cm)) for b in bearings]
    center = cm + 0.5 * (min(dev) + max(dev))
    arc = site_arc if site_arc is not None else 0.5 * (max(dev) - min(dev)) + arc_margin
    return center, arc


def collect_demos(
    n_pre: int,
    max_steps: int,
    site_radius: float,
    episodes: int,
    noise: float,
    tail: int,
    n_nominal: int,
    seed: int,
    arc_margin: float,
    site_arc: float | None,
):
    """Pass 1: fly the autopilot (clean + perturbed approaches), keep the soft
    landings, and read the goal arc off their achieved bearings. Pass 2: replay
    each keeper with a site drawn from that arc, building buffer transitions
    and BC pairs.

    Deriving the arc from what actually landed guarantees the commanded goals
    are reachable -- guessing an arc risks training on goals no trajectory in
    the buffer ever demonstrates.
    """
    moon_radius = ToMoonEnv().moon.radius
    recorded, bearings = [], []
    for k in range(episodes):
        # the first flight is the clean one; the rest perturb the approach
        n = 0.0 if k == 0 else noise
        execs, cleans, _info, b, soft = fly_expert(n, tail, seed + k, 0.0, n_nominal)
        recorded.append((execs, cleans, soft))
        if soft:
            bearings.append(b)
    if not bearings:
        raise SystemExit("no soft expert landings collected; lower --noise")

    center, arc = derive_arc(bearings, site_arc, arc_margin)

    rng = np.random.default_rng(seed)
    all_trans, all_bc_obs, all_bc_act = [], [], []
    for execs, cleans, soft in recorded:
        site = center + rng.uniform(-arc, arc)
        t, bo, ba = replay_window(execs, cleans, site, n_pre, max_steps, moon_radius, site_radius)
        # every flight teaches BC: a crashed approach still carries the clean
        # command the autopilot would have given (DART-style corrective
        # labels). Only real landings seed the buffer, so HER relabels
        # successes and never learns a crash as a goal worth reaching.
        all_bc_obs.extend(bo)
        all_bc_act.extend(ba)
        if soft:
            all_trans.append(t)
    return all_trans, all_bc_obs, all_bc_act, center, arc, bearings, len(recorded)


# --- behavior cloning (#1: start the actor as a soft-lander) -----------------


def bc_pretrain(
    model: TD3,
    obs_list: list[dict],
    act: list[np.ndarray],
    epochs: int,
    batch: int = 256,
    lr: float = 1e-3,
) -> None:
    """Regress the deterministic actor onto the autopilot's clean commands.

    Without this the actor starts random, crashes every episode, and -- since
    a crash is -1 for every relabeled goal -- HER never sees a single success
    to learn from. BC is what makes the sparse gated reward learnable at all.
    """
    dev = model.device
    keys = list(obs_list[0].keys())
    obs_t = {k: th.as_tensor(np.stack([o[k] for o in obs_list]), device=dev) for k in keys}
    # the actor is tanh-squashed and 99% of expert throttle commands sit at
    # exactly +-1, which tanh can only approach asymptotically; clipping the
    # targets keeps the MSE gradient from stalling on saturated outputs
    act_t = th.as_tensor(np.clip(np.stack(act), -0.999, 0.999), device=dev)
    opt = th.optim.Adam(model.actor.parameters(), lr=lr)
    n = len(act_t)
    for epoch in range(epochs):
        perm = th.randperm(n, device=dev)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            pred = model.actor({k: v[idx] for k, v in obs_t.items()})
            loss = th.nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  bc epoch {epoch + 1}/{epochs}: mse {total / n:.6f}")
    model.actor_target.load_state_dict(model.actor.state_dict())


# --- evaluation ---------------------------------------------------------------


def evaluate(
    model_path: str,
    episodes: int,
    device: str,
    make_env,
    watch: bool = False,
    model: TD3BC | None = None,
) -> tuple[int, int]:
    env = make_env("human" if watch else None, seed=12345)
    if model is None:
        # HerReplayBuffer is in the saved kwargs and SB3 refuses to rebuild it
        # without an env, so hand it the eval env; shrink the buffer too, since
        # loading otherwise allocates the full training-size replay arrays for
        # a run that never stores a transition
        model = TD3BC.load(model_path, env=env, device=device, custom_objects={"buffer_size": 1})
    hits = softs = 0
    misses = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        soft = landed_soft(info)
        softs += soft
        hit = bool(info.get("is_success"))
        hits += hit
        miss_km = info["moon_site_distance"] / 1000.0
        if soft:
            misses.append(miss_km)
        print(
            f"  ep {ep + 1:2d}: {info['termination']:<16} "
            f"site {math.degrees(info['site_angle']):+7.2f} deg  "
            f"rel {info['moon_rel_speed']:6.1f} m/s  miss {miss_km:7.1f} km"
            f"{'  ON SITE' if hit else ('  soft' if soft else '')}"
        )
    env.close()
    med = f"{np.median(misses):.1f} km" if misses else "n/a"
    print(f"  -> on-site {hits}/{episodes}  soft {softs}/{episodes}  median soft miss {med}")
    return hits, softs


# --- training -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="TD3 env steps")
    ap.add_argument(
        "--handoff",
        type=float,
        default=0.80,
        help="expert prefix fraction flown before the policy (the "
        "approach); the policy trains the landing window after",
    )
    ap.add_argument(
        "--site-arc",
        type=float,
        default=None,
        help="half-width of the random site arc, rad (default: "
        "derived from the demos' achieved bearings)",
    )
    ap.add_argument("--arc-margin", type=float, default=0.02, help="rad added to the derived arc")
    ap.add_argument(
        "--site-radius",
        type=float,
        default=None,
        help="on-site success radius, m (default: env's 8 km)",
    )
    ap.add_argument(
        "--max-steps", type=int, default=1200, help="policy-step cap per episode after the handoff"
    )
    ap.add_argument(
        "--demos",
        type=int,
        default=30,
        help="expert flights to fly; all of them teach BC, the "
        "ones that land softly (~30%%) seed the HER buffer",
    )
    ap.add_argument(
        "--orbit",
        action="store_true",
        help="orbit-first: the autopilot parks in a low moon orbit, "
        "then the policy flies deorbit+descent+landing DIRECT "
        "(no residual). Site targeting becomes a deorbit-timing "
        "problem RL can actually learn; goals span the whole "
        "moon. This is the recommended setup.",
    )
    ap.add_argument(
        "--residual",
        action="store_true",
        help="learn a bounded correction to the autopilot instead "
        "of replacing it; a zero action IS the autopilot, so "
        "training starts from a soft landing",
    )
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="how much of the action range the correction may move",
    )
    ap.add_argument(
        "--no-hover",
        action="store_true",
        help="orbit mode: use raw throttle instead of hover-relative "
        "control (a[0]=0 = hover). Hover control is on by "
        "default for --orbit; the landing throttle band is a "
        "razor-thin sliver around hover in raw terms",
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=None,
        help="demo noise (default 0.05 residual / 0.01 direct); in "
        "direct mode it is applied only before the last "
        "--noise-tail steps, since the descent has no margin",
    )
    ap.add_argument("--noise-tail", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument(
        "--critic-warmup",
        type=int,
        default=3000,
        help="critic-only gradient steps on the seeded demos before the actor is allowed to move",
    )
    ap.add_argument(
        "--bc-alpha",
        type=float,
        default=2.5,
        help="TD3+BC RL/BC balance; lower holds tighter to the "
        "demonstrations (0 = pure behavior cloning)",
    )
    ap.add_argument(
        "--explore-noise",
        type=float,
        default=None,
        help="TD3 exploration sigma (default 0.05 residual / 0.005 "
        "direct); the touchdown tolerates very little raw "
        "action noise, and the residual scales it down",
    )
    ap.add_argument(
        "--reward-scale",
        type=float,
        default=5.0,
        help="magnitude of the terminal bonus/penalty against a dense term bounded by 1",
    )
    ap.add_argument(
        "--descent-shape-w",
        type=float,
        default=None,
        help="weight of the dense soft-descent gradient (default "
        "2.0 for --orbit, 0 otherwise); makes 'stay landable' "
        "a followable gradient so the actor need not stumble "
        "onto a soft touchdown",
    )
    ap.add_argument(
        "--policy-delay",
        type=int,
        default=2,
        help="critic updates per actor update; raise to let the "
        "critic settle before the actor chases it",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="the landing window is ~500 steps, so at 0.99 the "
        "terminal reward is discounted to ~0.007 by the start "
        "of the window and only the dense term is visible",
    )
    ap.add_argument(
        "--bc-weight",
        type=float,
        default=None,
        help="weight on the BC term during training (default 0 "
        "residual / 1 direct); in residual mode the demos are "
        "zero residuals, so any weight here penalizes acting",
    )
    ap.add_argument(
        "--bc-lambda-max",
        type=float,
        default=10.0,
        help="ceiling on the RL/BC balance term; only meant to "
        "survive the untrained-critic startup, so keep it "
        "well above bc_alpha/mean|Q|",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate (default 3e-4 residual / 1e-4 direct; "
        "direct mode keeps it low to protect a fragile clone, "
        "while the residual bound protects the landing itself)",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--learning-starts", type=int, default=500)
    ap.add_argument("--n-sampled-goal", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument(
        "--watch",
        default=None,
        help="model to evaluate; opens a render window unless "
        "--no-render (headless jobs must pass it)",
    )
    ap.add_argument("--no-render", action="store_true", help="evaluate without opening a window")
    args = ap.parse_args()

    global SUCCESS_BONUS, FAILURE_PENALTY, DESCENT_SHAPE_W
    SUCCESS_BONUS = FAILURE_PENALTY = args.reward_scale
    if args.descent_shape_w is None:
        args.descent_shape_w = 2.0 if args.orbit else 0.0
    DESCENT_SHAPE_W = args.descent_shape_w
    # the residual scales the policy's authority down, so its noise budget is
    # correspondingly larger to land in the same effective range
    if args.noise is None:
        args.noise = 0.05 if args.residual else 0.01
    if args.explore_noise is None:
        args.explore_noise = 0.05 if args.residual else 0.005
    if args.lr is None:
        args.lr = 3e-4 if args.residual else 1e-4
    if args.bc_weight is None:
        args.bc_weight = 0.0 if args.residual else 1.0

    site_radius = args.site_radius
    if site_radius is None:
        site_radius = ToMoonEnv().moon_site_radius

    if args.orbit:
        # Orbit-first: the autopilot parks in a low moon orbit and lands on the
        # site from there; the policy learns the from-orbit segment (deorbit
        # timing + descent) direct, over the WHOLE moon.
        if args.max_steps < 2000:
            args.max_steps = 2500  # room to coast up to an orbit, then descend
        hover = not args.no_hover
        print(
            f"[1/5] caching the orbit-first expert (park -> deorbit -> land)"
            f"{'  [hover control]' if hover else '  [raw throttle]'}"
        )
        actions, n_pre, oinfo = orbit_expert(hover_control=hover)
        print(
            f"      expert episode {len(actions)} steps; parks in orbit at "
            f"{n_pre}; policy flies the last {len(actions) - n_pre}; "
            f"expert landed {oinfo['termination']} at "
            f"{oinfo['moon_rel_speed']:.1f} m/s"
        )
        center, arc = 0.0, math.pi  # goals span the entire moon
        sites = list(np.linspace(-math.pi, math.pi, args.demos, endpoint=False))
        print(f"[2/5] flying {args.demos} orbit demos across the whole moon")
        trans, bc_obs, bc_act, bearings, softs = collect_orbit_demos(
            n_pre, site_radius, args.max_steps, sites, seed=0, hover_control=hover
        )
        attempts = args.demos
        n_trans = sum(len(t) for t in trans)
        print(
            f"      {softs}/{attempts} landed softly; {n_trans} buffer "
            f"transitions, {len(bc_act)} BC pairs"
        )

        def make_env(render_mode=None, seed=None):
            return GoalMoonEnv(
                actions,
                0.0,
                center,
                arc,
                args.max_steps,
                site_radius=site_radius,
                render_mode=render_mode,
                seed=seed,
                n_pre=n_pre,
                env_max_steps=ORBIT_ENV_STEPS,
                hover_control=hover,
            )

        _run_training(args, make_env, trans, bc_obs, bc_act)
        return

    print("[1/5] caching the expert approach")
    actions = expert_actions()
    n_pre = int(args.handoff * len(actions))
    print(
        f"      expert episode {len(actions)} steps; handoff at {n_pre}; "
        f"policy flies the last {len(actions) - n_pre}; "
        f"success radius {site_radius / 1000:.1f} km"
    )

    mode = "residual" if args.residual else "direct"
    print(f"[2/5] collecting {args.demos} demos ({mode}, noise {args.noise})")
    if args.residual:
        # Pass 1: fly with a provisional arc purely to see where the autopilot
        # puts the touchdown, then set the goal arc to that measured span.
        prov_arc = args.site_arc if args.site_arc is not None else 0.35
        probe = ResidualGoalMoonEnv(
            actions,
            args.handoff,
            natural_touchdown_bearing(),
            prov_arc,
            args.max_steps,
            site_radius=site_radius,
            seed=0,
            residual_scale=args.residual_scale,
        )
        *_, bearings, softs = collect_residual_demos(probe, args.demos, args.noise, seed=0)
        if not bearings:
            raise SystemExit("no soft residual demos; lower --noise")
        # Center on where the autopilot actually puts it, but do NOT derive the
        # arc from the demo scatter: the approach is snapshotted, so zero-mean
        # residual noise cancels out and the touchdown barely moves (~0.1 deg),
        # which would make the goal arc degenerate and the task trivial. A
        # deliberate correction reaches much further than noise does -- a
        # constant bias alone shifts it ~0.5 deg while staying soft, and more
        # if spent at the right moments. So the arc is a stated target the
        # policy has to learn to reach, wider than the site radius on purpose.
        center = math.atan2(float(np.mean(np.sin(bearings))), float(np.mean(np.cos(bearings))))
        arc = args.site_arc if args.site_arc is not None else RESIDUAL_ARC
        # Pass 2: the real demos, with goals drawn from the reachable arc
        probe.set_site_arc(center, arc)
        trans, bc_obs, bc_act, bearings, softs = collect_residual_demos(
            probe, args.demos, args.noise, seed=1
        )
        probe.close()
        attempts = args.demos
    else:
        trans, bc_obs, bc_act, center, arc, bearings, attempts = collect_demos(
            n_pre,
            args.max_steps,
            site_radius,
            args.demos,
            args.noise,
            args.noise_tail,
            len(actions),
            0,
            args.arc_margin,
            args.site_arc,
        )
        softs = len(trans)
    n_trans = sum(len(t) for t in trans)
    print(
        f"      {softs}/{attempts} landed softly; {n_trans} buffer "
        f"transitions, {len(bc_act)} BC pairs"
    )
    print(
        f"      goal arc derived from touchdowns: center "
        f"{math.degrees(center):+.2f} deg +/- {math.degrees(arc):.2f} deg "
        f"({arc * ToMoonEnv().moon.radius / 1000:.1f} km of surface)"
    )

    def make_env(render_mode=None, seed=None):
        if args.residual:
            return ResidualGoalMoonEnv(
                actions,
                args.handoff,
                center,
                arc,
                args.max_steps,
                site_radius=site_radius,
                render_mode=render_mode,
                seed=seed,
                residual_scale=args.residual_scale,
            )
        return GoalMoonEnv(
            actions,
            args.handoff,
            center,
            arc,
            args.max_steps,
            site_radius=site_radius,
            render_mode=render_mode,
            seed=seed,
        )

    _run_training(args, make_env, trans, bc_obs, bc_act)


def _run_training(args, make_env, trans, bc_obs, bc_act) -> None:
    """Seed the HER buffer, behavior-clone the actor, warm the critic, then run
    TD3+BC. Shared by every mode (orbit / direct / residual); or, with --eval,
    just roll the saved policy."""
    if args.eval:
        model_path = args.watch or args.out
        if not model_path:
            raise SystemExit("--eval needs a model: --watch models/<run>.zip")
        evaluate(
            model_path,
            args.eval,
            args.device,
            make_env,
            watch=bool(args.watch) and not args.no_render,
        )
        return

    n_trans = sum(len(t) for t in trans)
    vec = DummyVecEnv([lambda: Monitor(make_env())])
    n_act = int(vec.action_space.shape[0])

    print("[3/5] building TD3 + HER (goal-conditioned, hindsight relabeling)")
    model = TD3BC(
        "MultiInputPolicy",
        vec,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=args.n_sampled_goal,
            goal_selection_strategy="future",
            copy_info_dict=True,  # REQUIRED: the landed_soft gate lives in info
        ),
        action_noise=NormalActionNoise(np.zeros(n_act), args.explore_noise * np.ones(n_act)),
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=1,
        gradient_steps=1,
        policy_delay=args.policy_delay,
        device=args.device,
        seed=0,
        verbose=0,
    )
    model.bc_alpha = args.bc_alpha
    model.bc_lambda_max = args.bc_lambda_max
    model.bc_weight = args.bc_weight

    for episode in trans:
        for o, no, a, r, d, i in episode:
            model.replay_buffer.add(
                {k: v[None] for k, v in o.items()},
                {k: v[None] for k, v in no.items()},
                a[None],
                np.array([r], np.float32),
                np.array([d]),
                [i],
            )
    n_success = sum(1 for e in trans for t in e if t[3] > 0.0)
    print(f"      seeded {n_trans} transitions ({n_success} already on-site before relabeling)")

    print(f"[4/5] behavior cloning the actor, {args.bc_epochs} epochs")
    bc_pretrain(model, bc_obs, bc_act, args.bc_epochs, lr=args.bc_lr)
    model.set_demos(bc_obs, bc_act)  # keep the anchor on the demos, not the buffer
    print("      BC policy, flown solo on random goals:")
    evaluate(None, 5, args.device, make_env, model=model)

    if args.critic_warmup > 0:
        print(f"      warming the critic on the seeded landings ({args.critic_warmup} steps)")
        model.warmup_critic(args.critic_warmup, args.batch_size)

    print(f"[5/5] TD3+BC + HER, {args.steps:,} steps")
    out = args.out or time.strftime("models/her_moon_%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    remaining = args.steps
    chunk = args.eval_every if args.eval_every > 0 else args.steps
    done_steps = 0
    while remaining > 0:
        n = min(chunk, remaining)
        model.learn(total_timesteps=n, reset_num_timesteps=False, progress_bar=False)
        remaining -= n
        done_steps += n
        model.save(out)
        if args.eval_every > 0 and remaining >= 0:
            print(f"  [eval @ {done_steps:,} steps]")
            evaluate(None, args.eval_episodes, args.device, make_env, model=model)
    model.save(out)
    print(
        f"saved {out}.zip -- evaluate with:\n"
        f"  python scripts/train_her.py --eval 20 --watch {out}.zip"
    )


if __name__ == "__main__":
    main()
