"""Fly the full ToMoon mission with a trained landing policy.

The scripted autopilot flies launch -> orbit -> injection -> capture into the
lunar parking orbit (site-independent, cached in the state bank); the policy
takes over at parking-orbit entry and flies deorbit, descent and touchdown.
Reports the env's own termination and return, i.e. the mission as the
environment scores it, not the training reward.

    python scripts/fly_moon_policy.py models/<run>/best.zip                  # 16 sites
    python scripts/fly_moon_policy.py models/<run>/best.zip --site 45 --render
    python scripts/fly_moon_policy.py models/<run>/best.zip --site 45 --gif landing.gif
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from orbital_gym.rl.moon_landing import MoonLandingEnv, load_bank


def fly(
    model,
    site_deg: float,
    extra_obs: bool,
    throttle_map: str,
    render: bool = False,
    gif: str | None = None,
    every: int = 4,
):
    bank = load_bank(verbose=False)
    env = MoonLandingEnv(
        bank,
        nominal=True,
        state_noise=0.0,
        extra_obs=extra_obs,
        throttle_map=throttle_map,
        render_mode="human" if render else ("rgb_array" if gif else None),
    )
    u = env.env
    frames = []
    # 1. the scripted prefix, from the pad, in the raw env
    u.reset(options={"site_angle": math.radians(site_deg)})
    for i, a in enumerate(bank.prefix_actions):
        _, _, term, trunc, info = u.step(a)
        if gif and i % (every * 8) == 0:
            frames.append(u.render())
        assert not (term or trunc), "prefix ended early"
    prefix_return = u._episode_return
    # 2. hand over: the wrapper's bookkeeping, the policy's actions
    env.start_ttl = bank.max_ttl
    env._t, env._cap = 0, env.window
    env._phi = env._potential()
    env._dv0 = u._dv_used
    obs = env._obs()
    done, n = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        n += 1
        if render:
            u.render()
        if gif and n % every == 0:
            frames.append(u.render())
    if gif and frames:
        import imageio

        frames = [f for f in frames if f is not None]
        imageio.mimsave(gif, frames, duration=1 / 20, loop=0)
        print(f"wrote {gif} ({len(frames)} frames)")
    env.close()
    return dict(
        site=site_deg,
        termination=info["termination"],
        rel=info["moon_rel_speed"],
        miss_km=info["moon_site_distance"] / 1000,
        policy_steps=n,
        native_total=u._episode_return,
        prefix_return=prefix_return,
        dv_used=info["dv_used"],
        dv_left=info["dv_remaining"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument(
        "--site", type=float, default=None, help="site angle, deg (default: 16-site grid)"
    )
    ap.add_argument("--sites", type=int, default=16)
    ap.add_argument("--no-extra-obs", action="store_true")
    ap.add_argument("--throttle-map", default="exp")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--gif", default=None)
    args = ap.parse_args()
    model = PPO.load(args.model, device="cpu")
    sites = (
        [args.site]
        if args.site is not None
        else list(np.degrees(np.linspace(-math.pi, math.pi, args.sites, endpoint=False)))
    )
    rows = []
    for s in sites:
        r = fly(
            model,
            s,
            not args.no_extra_obs,
            args.throttle_map,
            args.render,
            args.gif if len(sites) == 1 else None,
        )
        rows.append(r)
        print(
            f"site {r['site']:+7.1f} deg: {r['termination']:<16} rel {r['rel']:6.1f} m/s  "
            f"miss {r['miss_km']:6.1f} km  policy steps {r['policy_steps']:4d}  "
            f"mission return {r['native_total']:7.1f}  dv left {r['dv_left']:5.0f} m/s"
        )
    on = np.mean([r["termination"] == "landed_moon_site" for r in rows])
    soft = np.mean([r["termination"] in ("landed_moon_site", "landed_moon") for r in rows])
    print("-" * 72)
    print(
        f"on-site {on:.0%}  soft {soft:.0%}  mean mission return {np.mean([r['native_total'] for r in rows]):.0f}  "
        f"(autopilot alone: landed_moon_site, ~3040-3090)"
    )


if __name__ == "__main__":
    main()
