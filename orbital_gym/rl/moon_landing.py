"""The moon landing as an RL task: from a low lunar parking orbit to a soft
touchdown on a randomly placed site.

Why this slice. The scripted orbit-first autopilot (MoonAutopilot(orbit=True))
flies launch, orbit, trans-lunar injection and capture into a 60 km parking
orbit deterministically, and the dynamics never depend on where the site is.
Everything before parking-orbit entry is therefore a fixed, site-independent
prefix; everything after it -- choosing WHEN to deorbit and HOW to fly the
powered descent so the touchdown is soft and on the site -- is the part a
policy has to learn. This module poses exactly that.

What the wrapper changes relative to raw ToMoonEnv (none of it touches the
physics or the mission's own reward, which is still reported in info):

  * Start states come from a bank of expert states (every step of the
    autopilot's from-orbit segment, for many sites). A curriculum level in
    [0, 1] caps how far before touchdown an episode may begin, so training
    starts with "you are 50 m up, hovering over the site" and ends with
    "you are in orbit, the site is wherever". Expert states are reachable
    by construction; small state noise and site re-randomisation keep the
    policy from memorising the expert's exact tube.
  * Actions are in the MOON-local frame (hover-relative throttle, thrust
    direction as (up, east) relative to the moon). The env's own direction
    command is in the planet-local frame, which rotates with the rocket's
    position around the moon -- a pure coordinate nuisance for the network.
  * Reward is potential-based shaping on site distance and moon-relative
    speed plus a terminal outcome term. Shaping telescopes to a bounded total,
    so the terminal term decides what is optimal.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..autopilots import MoonAutopilot
from ..guidance import Guidance
from ..missions.to_moon import ToMoonEnv
from ..missions.to_orbit import _wrap_pi

ENV_MAX_STEPS = 12_000  # same horizon as the orbit-first expert flights
SNAP_SKIP = ("_renderer",)
POST_CAPTURE = ("moon_orbit", "deorbit", "pdescend")
BANK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache",
    "moon_bank.pkl",
)


def snapshot(env: ToMoonEnv) -> dict:
    d = {}
    for k, v in env.__dict__.items():
        if not k.startswith("_") or k in SNAP_SKIP:
            continue
        if k == "_trail":
            v = [(env._rx, env._ry)]
        elif k == "_debris":
            v = None
        d[k] = copy.deepcopy(v)
    return d


def restore(env: ToMoonEnv, snap: dict) -> None:
    env.__dict__.update(copy.deepcopy(snap))


@dataclass
class BankEntry:
    snap: dict
    ttl: int  # expert steps from this state to touchdown
    site: float  # site the expert was flying to
    mode: str  # expert mode at this state
    touch: float = 0.0  # body-frame bearing where that flight actually touched down
    act: object = None  # the expert's env-frame action at this state


@dataclass
class StateBank:
    entries: list = field(default_factory=list)
    prefix_actions: list = field(default_factory=list)  # launch -> orbit entry
    prefix_return: float = 0.0  # native return of the prefix
    handoff: int = 0  # step index of orbit entry
    orbit_entry: dict | None = None  # snapshot at the handoff
    descent_len: int = 0  # longest deorbit->touchdown
    max_ttl: int = 0

    def by_ttl(self):
        if not hasattr(self, "_by_ttl"):
            d: dict[int, list] = {}
            for e in self.entries:
                d.setdefault(e.ttl, []).append(e)
            self._by_ttl = d
            self._ttls = np.array(sorted(d))
        return self._by_ttl, self._ttls


def build_bank(n_sites: int = 24, coast_stride: int = 3, verbose: bool = True) -> StateBank:
    """Fly the orbit-first expert to n_sites sites; keep every from-orbit state."""
    bank = StateBank()
    sites = np.linspace(-math.pi, math.pi, n_sites, endpoint=False)
    for k, site in enumerate(sites):
        env = ToMoonEnv(max_episode_steps=ENV_MAX_STEPS, hover_control=True)
        _, info = env.reset(options={"site_angle": float(site)})
        pilot = MoonAutopilot(env, orbit=True)
        snaps, actions, handoff = [], [], None
        while True:
            s = snapshot(env)
            a = np.asarray(pilot(info), dtype=np.float32)
            if pilot.mode in POST_CAPTURE:
                if handoff is None:
                    handoff = env._steps
                    if bank.orbit_entry is None:
                        bank.orbit_entry = s
                        bank.handoff = handoff
                        bank.prefix_actions = list(actions)
                        bank.prefix_return = float(env._episode_return)
                snaps.append((s, pilot.mode, a))
            actions.append(a)
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        # the expert is soft for every site but misses a few by more than the
        # 8 km site radius; those flights still supply reachable states
        assert info["termination"] in ("landed_moon", "landed_moon_site"), (
            site,
            info["termination"],
        )
        assert handoff == bank.handoff, "orbit entry must be site-independent"
        n = len(snaps)
        # where this flight really came down (body frame): the expert lands
        # 6-8 km from the commanded site, which is the whole site radius, so
        # low-level starts aim at the touchdown itself and the site noise
        # grows with the level
        mx, my = env._moon_world()
        touch = _wrap_pi(math.atan2(env._ry - my, env._rx - mx) - env._moon_angle())
        first_deorbit = next(i for i, (_, m, _) in enumerate(snaps) if m != "moon_orbit")
        bank.descent_len = max(bank.descent_len, n - first_deorbit)
        for i, (s, mode, a) in enumerate(snaps):
            ttl = n - i
            if mode == "moon_orbit" and (i % coast_stride) and i != 0:
                continue  # coasting states are near-duplicates; thin them
            bank.entries.append(BankEntry(s, ttl, float(site), mode, touch, a))
            bank.max_ttl = max(bank.max_ttl, ttl)
        if verbose:
            print(
                f"  bank site {k + 1}/{n_sites} ({math.degrees(site):+6.1f} deg): "
                f"{n} from-orbit steps, descent {n - first_deorbit}, "
                f"touchdown {info['moon_rel_speed']:.1f} m/s, "
                f"{info['moon_site_distance'] / 1000:.1f} km off"
            )
        env.close()
    return bank


def load_bank(path: str = BANK_PATH, **kw) -> StateBank:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    bank = build_bank(**kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bank, f)
    return bank


# --- reward -------------------------------------------------------------------


@dataclass
class RewardConfig:
    w_dist: float = 2.0  # potential weight on site distance
    d_scale: float = 400_000.0  # m; ~orbit-to-far-side distance
    w_speed: float = 1.0  # potential weight on moon-relative speed
    v_scale: float = 1_100.0  # m/s; ~parking-orbit speed
    w_miss: float = 0.0  # potential weight on the predicted (zero-effort) miss
    miss_scale: float = 20_000.0  # m; the site-distance potential is dominated by
    # altitude during a descent, so lateral error gets
    # almost no gradient without this term (v4: policy
    # used a prior on the site's usual place instead
    # of reading the gauge)
    miss_cap: float = 500_000.0  # m; the value used while the coasting path does not
    # reach the surface (periapsis-based "miss" is +-450 km
    # garbage in a near-circular orbit; a constant there
    # makes the first retro impulses reward-neutral, not a dip)
    on_site: float = 10.0  # soft touchdown within the site radius
    soft_base: float = 1.0  # soft touchdown anywhere...
    soft_near: float = 4.0  # ...plus this * exp(-miss / soft_scale)
    soft_scale: float = 20_000.0
    crash_min: float = 4.0  # crash penalty at zero impact speed
    crash_max: float = 10.0  # crash penalty at >= crash_v_ref
    crash_v_ref: float = 300.0
    fail: float = 10.0  # escape / window limit / time limit
    fuel: float = 0.0  # per m/s of delta-v spent
    hover_alt: float = 500.0  # m; below this altitude every step costs...
    hover_cost: float = 0.0  # ...this much (gamma 0.9995 makes dithering at the
    # surface nearly free: v9's mean policy hovered at
    # 2-5 m for 30-150 s shaving a miss already inside
    # the site radius, and drifted off-site doing it)


# the reward of the reference result (run ppo-zem-v15, EXPERIMENTS.md); the
# registered practice id and the trainer's defaults both read it
REFERENCE_REWARD = RewardConfig(w_miss=2.0, fuel=0.0005, hover_cost=0.02, hover_alt=500.0)
REFERENCE_GAMMA = 0.9995


class MoonLandingEnv(gym.Env):
    """Parking orbit -> soft touchdown on a random site. See module docstring."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_EXTRA = Guidance.N_CHANNELS  # zero-effort guidance channels, see Guidance

    def __init__(
        self,
        bank: StateBank | str = BANK_PATH,
        level: float = 0.0,
        reward: RewardConfig | None = None,
        gamma: float = 0.995,
        window: int = 1_800,
        window_slack: int = 400,
        cap_mult: float = 1.0,
        state_noise: float = 1.0,
        site_noise: float = 1.0,
        site_sigma_max: float = 0.04,
        nominal: bool = False,
        throttle_map: str = "lin",
        extra_obs: bool = False,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.bank = load_bank(bank) if isinstance(bank, str) else bank
        self.level = float(level)
        self.rc = copy.copy(reward) if reward is not None else RewardConfig()
        self.gamma = gamma
        self.window = window
        self.window_slack = window_slack
        self.cap_mult = cap_mult
        self.state_noise = state_noise
        self.site_noise = site_noise
        # cap on the descent-start site offset (rad). The expert touches down
        # 13-48 km upstream of the ballistic impact point of its post-deorbit
        # states; a site placed at or beyond that point needs a stay-aloft
        # manoeuvre the real from-orbit task never requires (measured, v5:
        # 0-9% on-site for those vs 81-94% upstream). 0.04 rad = 6.4 km.
        self.site_sigma_max = site_sigma_max
        self.nominal = nominal  # eval mode: orbit-entry state, no noise
        self.throttle_map = throttle_map
        self.extra_obs = extra_obs
        self.render_mode = render_mode
        self.env = ToMoonEnv(
            max_episode_steps=ENV_MAX_STEPS, hover_control=True, render_mode=render_mode
        )
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        n = self.env.observation_space.shape[0] + 1 + (self.N_EXTRA if extra_obs else 0)
        self.observation_space = spaces.Box(-20.0, 20.0, (n,), np.float32)
        self._t = 0
        self._cap = window
        self._phi = 0.0
        self._dv0 = 0.0
        self.start_ttl = 0
        self._start_in_orbit = False

    # -- curriculum ----------------------------------------------------------

    def set_level(self, level: float) -> None:
        self.level = float(min(1.0, max(0.0, level)))

    def _ttl_cap(self) -> int:
        """How far before touchdown an episode may start, at this level.
        0 -> 15 steps (final metres); 0.5 -> the powered descent up to the
        end of the retro burn; 0.5-0.65 -> the retro burn itself and the
        last ~40 s of coast before it (the commit decision, spread over
        three levels instead of one); 1 -> anywhere in the parking orbit."""
        lam = self.level
        d, T = self.bank.descent_len, self.bank.max_ttl
        d0, d1 = d - 30, d + 40
        if lam <= 0.5:
            return round(15 + (d0 - 15) * (lam / 0.5) ** 1.5)
        if lam <= 0.65:
            return round(d0 + (d1 - d0) * (lam - 0.5) / 0.15)
        return round(d1 + (T - d1) * (lam - 0.65) / 0.35)

    def _sample_entry(self) -> BankEntry:
        by_ttl, ttls = self.bank.by_ttl()
        cap = self._ttl_cap()
        # half the episodes start in the newest 15% of the allowed range
        # (the states the level just opened; at 0.55 that is the retro-burn
        # commit, which a 40% band diluted to ~7% of episodes), the rest
        # anywhere below it (keeps the earlier skills rehearsed)
        if self._rng.random() < 0.5:
            t = self._rng.uniform(0.85 * cap, cap)
        else:
            t = self._rng.uniform(1, cap)
        idx = int(np.searchsorted(ttls, t))
        idx = min(max(idx, 0), len(ttls) - 1)
        cands = by_ttl[int(ttls[idx])]
        return cands[self._rng.integers(len(cands))]

    def _place(self) -> None:
        """Restore a start state (plus noise) and choose the site."""
        u = self.env
        if self.nominal:
            restore(u, self.bank.orbit_entry)
            site = self._rng.uniform(-math.pi, math.pi)
            self.start_ttl = self.bank.max_ttl
        else:
            e = self._sample_entry()
            restore(u, e.snap)
            self.start_ttl = e.ttl
            self._start_in_orbit = e.mode == "moon_orbit"
            lam = self.level
            if e.mode == "moon_orbit":
                site = self._rng.uniform(-math.pi, math.pi)
            else:
                # off-nominal site: a diversion whose size grows with level
                sig = self.site_noise * min(self.site_sigma_max, 0.002 + 0.12 * lam)
                site = e.touch + self._rng.normal(0.0, sig)
            if self.state_noise > 0.0:
                self._perturb(lam)
        u.set_site_angle(_wrap_pi(site))
        u._last_site_d = u._moon_site_distance()
        u._last_ang = math.atan2(u._ry, u._rx)
        u._episode_return = 0.0
        u._terminal = None
        u._debris = None
        u._trail = [(u._rx, u._ry)]

    def _perturb(self, lam: float) -> None:
        u = self.env
        mx, my = u._moon_world()
        _mvx, _mvy = u._moon_velocity()
        px, py = u._rx - mx, u._ry - my
        d = math.hypot(px, py)
        malt = d - u.moon.radius
        sv = self.state_noise * (0.5 + 15.0 * lam)  # m/s
        sp = self.state_noise * (0.02 * malt + 5.0 + 500.0 * lam)  # m
        u._vx += self._rng.normal(0.0, sv)
        u._vy += self._rng.normal(0.0, sv)
        # radial: keep clear of the ground; tangential: slide along the surface
        dr = self._rng.normal(0.0, 0.3 * sp)
        dt = self._rng.normal(0.0, sp)
        new_alt = max(20.0, malt + dr)
        ang = math.atan2(py, px) + dt / d
        r_new = u.moon.radius + new_alt
        u._rx = mx + r_new * math.cos(ang)
        u._ry = my + r_new * math.sin(ang)
        u._theta += self._rng.normal(0.0, 0.05 + 0.2 * lam)
        u._target_theta = u._theta
        # fuel: a little less or more than the expert had (never below 15%)
        s = u.rocket.stages[u._stage]
        f = u._fuel * (1.0 + self._rng.normal(0.0, 0.1 * lam))
        f = min(s.prop_mass, max(0.15 * s.prop_mass, f))
        u._mass += f - u._fuel
        u._fuel = f

    # -- gym API ---------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._place()
        u = self.env
        if options and "site_angle" in options:
            u.set_site_angle(float(options["site_angle"]))
            u._last_site_d = u._moon_site_distance()
        self._t = 0
        # cap_mult * expert time-to-touchdown + slack; an orbit start with a
        # re-randomised site may need a full orbit (~1300 steps) of waiting
        # first, so it gets the whole window (v8 capped those at ttl+400 and
        # taught "a far site is a guaranteed timeout")
        self._cap = (
            self.window
            if (self.nominal or self._start_in_orbit)
            else min(self.window, int(self.cap_mult * self.start_ttl) + self.window_slack)
        )
        self._phi = self._potential()
        self._dv0 = u._dv_used
        return self._obs(), self._info(u._info(), 0.0)

    def _guidance(self) -> Guidance:
        """The zero-effort landing instruments for the current state (one
        object serves both the observation channels and the potential)."""
        return Guidance.from_env(self.env, self.env._site_angle)

    def _potential(self) -> float:
        u, rc = self.env, self.rc
        mvx, mvy = u._moon_velocity()
        v = math.hypot(u._vx - mvx, u._vy - mvy)
        phi = -rc.w_dist * u._moon_site_distance() / rc.d_scale - rc.w_speed * v / rc.v_scale
        # asinh-shaped miss term; in orbit it is the stop-and-drop virtual
        # miss, which gives a continuous timing gradient while coasting toward
        # the right phase (v7-v10 held it constant there and the timing had to
        # come from the terminal reward alone, hundreds of steps away)
        return phi + self._guidance().miss_potential(rc.w_miss, rc.miss_scale)

    def _moon_frame(self):
        u = self.env
        mx, my = u._moon_world()
        px, py = u._rx - mx, u._ry - my
        d = max(1.0, math.hypot(px, py))
        ux, uy = px / d, py / d
        return (ux, uy), (uy, -ux)  # up, east (matches the obs convention)

    def _throttle_a0(self, a0: float) -> float:
        """Policy throttle channel -> the env's hover-relative a0.

        'exp' passes a0 straight through (throttle = hover * exp(3 a0)): a
        0.33 action noise is +-1 g of random vertical acceleration per second,
        which made stochastic landings crash and the mean policy hover-dither
        (ppo-curriculum-v2). 'lin' maps a0 <= 0 linearly from engine-off to
        hover and a0 > 0 quadratically from hover to full thrust, so the same
        noise near hover is a fraction of a g while full thrust stays reachable.
        """
        if self.throttle_map == "exp":
            return float(a0)
        u = self.env
        hover = max(1e-4, u.hover_throttle())
        t = hover * (1.0 + a0) if a0 <= 0.0 else hover + (1.0 - hover) * a0 * a0
        if t < 0.02 * hover:
            return -1.0  # engine-off deadzone
        return float(np.clip(math.log(t / hover) / u.HOVER_K, -1.0, 1.0))

    def expert_action(self, entry: BankEntry) -> np.ndarray:
        """The expert's action at a bank state, expressed in THIS wrapper's
        action space (moon frame, chosen throttle map). The env must currently
        hold that state (call right after restore)."""
        a = np.asarray(entry.act, dtype=np.float64)
        u = self.env
        hover = max(1e-4, u.hover_throttle())
        thr = 0.0 if a[0] <= -0.98 else min(1.0, hover * math.exp(a[0] * u.HOVER_K))
        if self.throttle_map == "exp":
            a0 = a[0]
        elif thr <= hover:
            a0 = thr / hover - 1.0
        else:
            a0 = math.sqrt((thr - hover) / max(1e-6, 1.0 - hover))
        pu_x, pu_y, pe_x, pe_y = u._local_frame()
        wx, wy = a[1] * pu_x + a[2] * pe_x, a[1] * pu_y + a[2] * pe_y
        up, east = self._moon_frame()
        return np.array(
            [np.clip(a0, -1, 1), wx * up[0] + wy * up[1], wx * east[0] + wy * east[1]],
            dtype=np.float32,
        )

    def _env_action(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).flatten()
        a[0] = self._throttle_a0(float(np.clip(a[0], -1.0, 1.0)))
        up, east = self._moon_frame()
        wx = a[1] * up[0] + a[2] * east[0]
        wy = a[1] * up[1] + a[2] * east[1]
        pu, pe = self.env._local_frame()[0:2], self.env._local_frame()[2:4]
        return np.array([a[0], wx * pu[0] + wy * pu[1], wx * pe[0] + wy * pe[1]], dtype=np.float32)

    def step(self, action):
        u, rc = self.env, self.rc
        dv_before = u._dv_used
        _, native_r, terminated, truncated, info = u.step(self._env_action(action))
        self._t += 1
        term = info["termination"]
        if not terminated and not truncated and self._t >= self._cap:
            truncated = True
            term = "window_limit"
            info = dict(info)
            info["termination"] = term
        done = terminated or truncated
        phi_new = 0.0 if done else self._potential()
        # difference form, not gamma*phi' - phi: with a negative potential the
        # discounted form pays (1-gamma)*|phi| per step just for staying alive
        # (measured: +15 of the expert's 24 shaped return was loitering credit)
        reward = phi_new - self._phi
        self._phi = phi_new
        reward -= rc.fuel * (u._dv_used - dv_before)
        if rc.hover_cost > 0.0 and not done and u._moon_distance() - u.moon.radius < rc.hover_alt:
            reward -= rc.hover_cost
        miss = info["moon_site_distance"]
        if term == "landed_moon_site":
            reward += rc.on_site
        elif term == "landed_moon":
            reward += rc.soft_base + rc.soft_near * math.exp(-miss / rc.soft_scale)
        elif term == "crashed":
            f = min(1.0, info["moon_rel_speed"] / rc.crash_v_ref)
            reward -= rc.crash_min + (rc.crash_max - rc.crash_min) * f
        elif done:
            reward -= rc.fail
        return self._obs(), float(reward), terminated, truncated, self._info(info, native_r)

    def _guidance_obs(self) -> np.ndarray:
        return self._guidance().channels()

    def _obs(self) -> np.ndarray:
        base = self.env._obs()
        left = (self._cap - self._t) / max(1, self.window)
        parts = [base, np.array([left], np.float32)]
        if self.extra_obs:
            parts.append(self._guidance_obs())
        return np.concatenate(parts)

    def _info(self, info: dict, native_r: float) -> dict:
        info = dict(info)
        info["native_reward"] = native_r
        info["native_return"] = self.env._episode_return
        info["start_ttl"] = self.start_ttl
        info["level"] = self.level
        info["on_site"] = info["termination"] == "landed_moon_site"
        info["success"] = bool(info["on_site"])
        info["soft"] = info["termination"] in ("landed_moon", "landed_moon_site")
        info["miss_km"] = info["moon_site_distance"] / 1000.0
        info["dv_spent"] = self.env._dv_used - self._dv0
        return info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class MoonLandingFromOrbit(MoonLandingEnv):
    """`MoonLandingFromOrbit-v0`: the landing task frozen at the reference
    configuration (decisions 0006 and 0008): hover-relative exponential
    throttle, the zero-effort guidance channels, the reference reward and
    its gamma. The knobs are the practice ones: `level` (0 = the last
    metres before touchdown, 1 = anywhere from parking-orbit entry),
    `site_noise` (how far the site is re-randomised from the expert's),
    `extra_obs` (drop the guidance channels for a harder observation) and
    `nominal` (every episode starts at parking-orbit entry, no noise: the
    evaluation distribution). The expert state bank is built on first use
    (about ten seconds) and cached under `cache/`."""

    def __init__(
        self,
        level: float = 1.0,
        site_noise: float = 1.0,
        extra_obs: bool = True,
        nominal: bool = False,
        render_mode: str | None = None,
        seed: int | None = None,
        bank: StateBank | str = BANK_PATH,
    ):
        super().__init__(
            bank,
            level=level,
            reward=REFERENCE_REWARD,
            gamma=REFERENCE_GAMMA,
            site_noise=site_noise,
            nominal=nominal,
            throttle_map="exp",
            extra_obs=extra_obs,
            render_mode=render_mode,
            seed=seed,
        )
