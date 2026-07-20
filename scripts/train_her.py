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

import numpy as np

try:
    import torch as th
    from torch.nn import functional as F
    from stable_baselines3 import TD3, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import polyak_update
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    raise SystemExit("stable-baselines3 not installed; run: pip install -e '.[train]'")

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_gym.missions.to_moon import ToMoonEnv
from autopilot_demo import MoonAutopilot

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
        return math.atan2(math.sin(u._site_angle + miss),
                          math.cos(u._site_angle + miss))
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


SUCCESS_BONUS = 50.0
FAILURE_PENALTY = 50.0


def goal_arc(achieved: np.ndarray, desired: np.ndarray,
             moon_radius: float) -> np.ndarray:
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


def goal_reward(achieved: np.ndarray, desired: np.ndarray, infos: list,
                moon_radius: float, site_radius: float) -> np.ndarray:
    """Vectorized HER reward, dense in the goal.

    Per step: -arc/(pi*R) in [-1, 0], the normalized surface distance between
    where this trajectory is headed and the commanded site. Because it is a
    function of the goal at EVERY step, relabeling a whole trajectory produces
    a whole trajectory of informative rewards -- a purely terminal criterion
    gave HER one usable transition in ~480 (0.15%), which is not learnable.

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
    reward = -arc / (math.pi * moon_radius)
    soft = np.array([bool(i.get("landed_soft", False)) for i in infos])
    # a terminal step is any step the env labeled; non-terminal steps have None
    ended = np.array([bool(i.get("termination")) for i in infos])
    reward = reward + SUCCESS_BONUS * (soft & (arc <= site_radius))
    reward = reward - FAILURE_PENALTY * (ended & ~soft)
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
    # (measured: collapse by 3k steps even WITH the BC term). Capping it keeps
    # the anchor meaningful early; once Q reaches its true scale (~50 here, the
    # terminal bonus) lambda falls to ~0.05 and the cap stops binding.
    bc_lambda_max = 1.0

    def warmup_critic(self, gradient_steps: int, batch_size: int = 512) -> None:
        """Fit the critic on the seeded buffer before the actor may move.

        The actor is a behavior-cloned soft-lander at this point; letting it
        chase an untrained critic throws that away. This gives Q a real scale
        first, using only demonstration data.
        """
        self.policy.set_training_mode(True)
        for step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None \
                else self.gamma
            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                next_q = th.cat(self.critic_target(replay_data.next_observations,
                                                   next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                target_q = replay_data.rewards + \
                    (1 - replay_data.dones) * discounts * next_q
            current_q = self.critic(replay_data.observations, replay_data.actions)
            loss = sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)
            if (step + 1) % max(1, gradient_steps // 5) == 0:
                with th.no_grad():
                    scale = self.critic.q1_forward(
                        replay_data.observations, replay_data.actions).abs().mean()
                print(f"  critic warmup {step + 1}/{gradient_steps}: "
                      f"loss {loss.item():.4f}  mean|Q| {scale.item():.2f}")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, bc_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None \
                else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations)
                                + noise).clamp(-1, 1)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + \
                    (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations,
                                           replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values)
                              for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                pi = self.actor(replay_data.observations)
                q_pi = self.critic.q1_forward(replay_data.observations, pi)
                lmbda = th.clamp(self.bc_alpha / q_pi.abs().mean().detach(),
                                 max=self.bc_lambda_max)
                bc_loss = F.mse_loss(pi, replay_data.actions)
                actor_loss = -lmbda * q_pi.mean() + bc_loss
                actor_losses.append(actor_loss.item())
                bc_losses.append(bc_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(),
                              self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(),
                              self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats,
                              self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats,
                              self.actor_batch_norm_stats_target, 1.0)

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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    SNAP_SKIP = ("_renderer",)

    def __init__(self, actions: list[np.ndarray], handoff: float,
                 site_center: float, site_arc: float, max_steps: int,
                 site_radius: float | None = None, render_mode: str | None = None,
                 seed: int | None = None):
        super().__init__()
        self.env = ToMoonEnv(render_mode=render_mode)
        u = self.env
        self._actions = actions
        self._n_pre = int(handoff * len(actions))
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
        self.observation_space = spaces.Dict({
            "observation": u.observation_space,
            "achieved_goal": spaces.Box(-1.0, 1.0, (GOAL_DIM,), np.float32),
            "desired_goal": spaces.Box(-1.0, 1.0, (GOAL_DIM,), np.float32),
        })

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
        self._snap = {k: copy.deepcopy(v) for k, v in u.__dict__.items()
                      if k.startswith("_") and k not in self.SNAP_SKIP}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._ensure_snapshot()
        u = self.env
        u.__dict__.update(copy.deepcopy(self._snap))
        site = float(options["site_angle"]) if options and "site_angle" in options \
            else self.site_center + self._rng.uniform(-self.site_arc, self.site_arc)
        u.set_site_angle(site)
        u._last_site_d = u._moon_site_distance()
        self._desired = _desired_cossin(u)
        self._t = 0
        info = dict(u._info())
        info["landed_soft"] = False
        return goal_obs(u), info

    def step(self, action):
        u = self.env
        _, _, terminated, truncated, info = self.env.step(action)
        self._t += 1
        info = dict(info)
        if self._t >= self.max_steps and not terminated:
            # the window cap is the wrapper's, not the env's, so the env never
            # sets a terminal label for it -- name it rather than leave None
            truncated = True
            info["termination"] = info.get("termination") or "window_limit"
        info["landed_soft"] = landed_soft(info)
        gobs = goal_obs(u)
        reward = float(goal_reward(gobs["achieved_goal"], self._desired, [info],
                                   self.moon_radius, self.site_radius)[0])
        info["is_success"] = bool(goal_success(gobs["achieved_goal"], self._desired,
                                               [info], self.moon_radius,
                                               self.site_radius)[0])
        return gobs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER calls this with batched goals and a list of stored infos.
        infos = info if isinstance(info, (list, np.ndarray)) else [info]
        return goal_reward(achieved_goal, desired_goal, infos,
                           self.moon_radius, self.site_radius)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# --- expert flights ----------------------------------------------------------

def fly_expert(noise: float = 0.0, tail: int = 300, seed: int = 0,
               site_angle: float = 0.0, n_nominal: int | None = None):
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
            a = np.clip(clean + rng.normal(0.0, noise, clean.shape),
                        -1.0, 1.0).astype(np.float32)
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


def replay_window(exec_acts, clean_acts, site: float, n_pre: int, max_steps: int,
                  moon_radius: float, site_radius: float):
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
            r = float(goal_reward(nxt["achieved_goal"], nxt["desired_goal"], [info],
                                  moon_radius, site_radius)[0])
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


def collect_demos(n_pre: int, max_steps: int, site_radius: float, episodes: int,
                  noise: float, tail: int, n_nominal: int, seed: int,
                  arc_margin: float, site_arc: float | None):
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
        execs, cleans, info, b, soft = fly_expert(n, tail, seed + k, 0.0, n_nominal)
        recorded.append((execs, cleans, soft))
        if soft:
            bearings.append(b)
    if not bearings:
        raise SystemExit("no soft expert landings collected; lower --noise")

    # The goal arc is the span the demos actually landed in -- wrap-safe, and
    # tight rather than generous: commanding goals no trajectory in the buffer
    # ever reached would train against targets nothing demonstrates.
    cm = math.atan2(float(np.mean(np.sin(bearings))), float(np.mean(np.cos(bearings))))
    dev = [math.atan2(math.sin(b - cm), math.cos(b - cm)) for b in bearings]
    center = cm + 0.5 * (min(dev) + max(dev))
    arc = site_arc if site_arc is not None else 0.5 * (max(dev) - min(dev)) + arc_margin

    rng = np.random.default_rng(seed)
    all_trans, all_bc_obs, all_bc_act = [], [], []
    for execs, cleans, soft in recorded:
        site = center + rng.uniform(-arc, arc)
        t, bo, ba = replay_window(execs, cleans, site, n_pre, max_steps,
                                  moon_radius, site_radius)
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

def bc_pretrain(model: TD3, obs_list: list[dict], act: list[np.ndarray],
                epochs: int, batch: int = 256, lr: float = 1e-3) -> None:
    """Regress the deterministic actor onto the autopilot's clean commands.

    Without this the actor starts random, crashes every episode, and -- since
    a crash is -1 for every relabeled goal -- HER never sees a single success
    to learn from. BC is what makes the sparse gated reward learnable at all.
    """
    dev = model.device
    keys = list(obs_list[0].keys())
    obs_t = {k: th.as_tensor(np.stack([o[k] for o in obs_list]), device=dev)
             for k in keys}
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
            idx = perm[i:i + batch]
            pred = model.actor({k: v[idx] for k, v in obs_t.items()})
            loss = th.nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  bc epoch {epoch + 1}/{epochs}: mse {total / n:.6f}")
    model.actor_target.load_state_dict(model.actor.state_dict())


# --- evaluation ---------------------------------------------------------------

def evaluate(model_path: str, episodes: int, device: str, handoff: float,
             center: float, arc: float, max_steps: int, site_radius: float,
             actions: list[np.ndarray], watch: bool = False,
             model: TD3BC | None = None) -> tuple[int, int]:
    render = "human" if watch else None
    env = GoalMoonEnv(actions, handoff, center, arc, max_steps,
                      site_radius=site_radius, render_mode=render, seed=12345)
    if model is None:
        model = TD3BC.load(model_path, device=device)
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
        print(f"  ep {ep + 1:2d}: {info['termination']:<16} "
              f"site {math.degrees(info['site_angle']):+7.2f} deg  "
              f"rel {info['moon_rel_speed']:6.1f} m/s  miss {miss_km:7.1f} km"
              f"{'  ON SITE' if hit else ('  soft' if soft else '')}")
    env.close()
    med = f"{np.median(misses):.1f} km" if misses else "n/a"
    print(f"  -> on-site {hits}/{episodes}  soft {softs}/{episodes}  "
          f"median soft miss {med}")
    return hits, softs


# --- training -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="TD3 env steps")
    ap.add_argument("--handoff", type=float, default=0.80,
                    help="expert prefix fraction flown before the policy (the "
                         "approach); the policy trains the landing window after")
    ap.add_argument("--site-arc", type=float, default=None,
                    help="half-width of the random site arc, rad (default: "
                         "derived from the demos' achieved bearings)")
    ap.add_argument("--arc-margin", type=float, default=0.02,
                    help="rad added to the derived arc")
    ap.add_argument("--site-radius", type=float, default=None,
                    help="on-site success radius, m (default: env's 8 km)")
    ap.add_argument("--max-steps", type=int, default=1200,
                    help="policy-step cap per episode after the handoff")
    ap.add_argument("--demos", type=int, default=30,
                    help="expert flights to fly; all of them teach BC, the "
                         "ones that land softly (~30%%) seed the HER buffer")
    ap.add_argument("--noise", type=float, default=0.01,
                    help="demo approach noise; applied only before the last "
                         "--noise-tail steps (the descent has no margin)")
    ap.add_argument("--noise-tail", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--critic-warmup", type=int, default=3000,
                    help="critic-only gradient steps on the seeded demos "
                         "before the actor is allowed to move")
    ap.add_argument("--bc-alpha", type=float, default=2.5,
                    help="TD3+BC RL/BC balance; lower holds tighter to the "
                         "demonstrations (0 = pure behavior cloning)")
    ap.add_argument("--explore-noise", type=float, default=0.005,
                    help="TD3 exploration sigma; the touchdown tolerates very "
                         "little (0.01 never soft-lands)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="TD3 learning rate (low to protect the BC init)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--learning-starts", type=int, default=500)
    ap.add_argument("--n-sampled-goal", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--watch", default=None)
    args = ap.parse_args()

    print("[1/5] caching the expert approach")
    actions = expert_actions()
    n_pre = int(args.handoff * len(actions))
    site_radius = args.site_radius
    if site_radius is None:
        site_radius = ToMoonEnv().moon_site_radius
    print(f"      expert episode {len(actions)} steps; handoff at {n_pre}; "
          f"policy flies the last {len(actions) - n_pre}; "
          f"success radius {site_radius / 1000:.1f} km")

    print(f"[2/5] flying {args.demos} expert demos "
          f"(approach noise {args.noise}, clean tail {args.noise_tail})")
    trans, bc_obs, bc_act, center, arc, bearings, attempts = collect_demos(
        n_pre, args.max_steps, site_radius, args.demos, args.noise,
        args.noise_tail, len(actions), 0, args.arc_margin, args.site_arc)
    n_trans = sum(len(t) for t in trans)
    print(f"      {len(trans)}/{attempts} landed softly; {n_trans} buffer "
          f"transitions, {len(bc_act)} BC pairs (all flights)")
    print(f"      goal arc derived from touchdowns: center "
          f"{math.degrees(center):+.2f} deg +/- {math.degrees(arc):.2f} deg "
          f"({arc * ToMoonEnv().moon.radius / 1000:.1f} km of surface)")

    if args.eval:
        model_path = args.watch or args.out
        if not model_path:
            raise SystemExit("--eval needs a model: --watch models/<run>.zip")
        evaluate(model_path, args.eval, args.device, args.handoff, center, arc,
                 args.max_steps, site_radius, actions, watch=bool(args.watch))
        return

    def _thunk():
        return Monitor(GoalMoonEnv(actions, args.handoff, center, arc,
                                   args.max_steps, site_radius=site_radius))
    vec = DummyVecEnv([_thunk])

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
        action_noise=NormalActionNoise(np.zeros(2), args.explore_noise * np.ones(2)),
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=1,
        gradient_steps=1,
        device=args.device,
        seed=0,
        verbose=0,
    )
    model.bc_alpha = args.bc_alpha

    for episode in trans:
        for o, no, a, r, d, i in episode:
            model.replay_buffer.add(
                {k: v[None] for k, v in o.items()},
                {k: v[None] for k, v in no.items()},
                a[None], np.array([r], np.float32), np.array([d]), [i])
    n_success = sum(1 for e in trans for t in e if t[3] > 0.0)
    print(f"      seeded {n_trans} transitions ({n_success} already on-site "
          f"before relabeling)")

    print(f"[4/5] behavior cloning the actor, {args.bc_epochs} epochs")
    bc_pretrain(model, bc_obs, bc_act, args.bc_epochs, lr=args.bc_lr)
    print("      BC policy, flown solo on random goals:")
    evaluate(None, 5, args.device, args.handoff, center, arc, args.max_steps,
             site_radius, actions, model=model)

    if args.critic_warmup > 0:
        print(f"      warming the critic on the seeded landings "
              f"({args.critic_warmup} steps)")
        model.warmup_critic(args.critic_warmup, args.batch_size)

    print(f"[5/5] TD3+BC + HER, {args.steps:,} steps")
    out = args.out or time.strftime("models/her_moon_%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    remaining = args.steps
    chunk = args.eval_every if args.eval_every > 0 else args.steps
    done_steps = 0
    while remaining > 0:
        n = min(chunk, remaining)
        model.learn(total_timesteps=n, reset_num_timesteps=False,
                    progress_bar=False)
        remaining -= n
        done_steps += n
        model.save(out)
        if args.eval_every > 0 and remaining >= 0:
            print(f"  [eval @ {done_steps:,} steps]")
            evaluate(None, args.eval_episodes, args.device, args.handoff, center,
                     arc, args.max_steps, site_radius, actions, model=model)
    model.save(out)
    print(f"saved {out}.zip -- evaluate with:\n"
          f"  python scripts/train_her.py --eval 20 --watch {out}.zip")


if __name__ == "__main__":
    main()
