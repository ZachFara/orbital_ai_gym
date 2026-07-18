"""Train PPO on the orbital_gym missions with Stable-Baselines3.

    pip install -e ".[train]"
    python scripts/train_ppo.py                          # to-orbit, 1M steps
    python scripts/train_ppo.py --mission return         # orbit-and-return
    python scripts/train_ppo.py --device mps            # force Apple GPU
    python scripts/train_ppo.py --out models/my_run      # named run
    python scripts/train_ppo.py --eval 5 --watch models/<run>.zip
    python scripts/train_ppo.py --watch models/<run>.zip # fly it in the UI

Every run saves under a fresh name (timestamped unless --out is given), so
training never overwrites an older model.

Trains WITHOUT terminate_on_orbit: episodes run to the time limit, so the
policy must not only reach orbit but hold it for the rest of the episode.
(The terminate-on-orbit shortcut is reward-equivalent but produces policies
that have never seen a post-orbit state and fly themselves back into the
ground once one appears.) Episode returns for orientation: ~0 tumble/crash,
~100-900 suborbital lobs, ~4000 loitering on a wild barely-bound ellipse,
~7500+ reaching a *circular* orbit and keeping it to the time limit.
"""

from __future__ import annotations

import argparse
import math
import os
import time

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
except ImportError:
    raise SystemExit("stable-baselines3 not installed; run: pip install -e '.[train]'")

from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv
from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv

MISSIONS = {"orbit": TwoStageToOrbitEnv, "return": OrbitAndReturnEnv}

GAMMA = 0.999  # long horizon: the duration reward arrives late


def evaluate(model_path: str, episodes: int, device: str, mission: str) -> None:
    env = MISSIONS[mission]()
    model = PPO.load(model_path, device=device)
    successes = 0
    returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        space_s = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            space_s += env.step_duration if info["in_space"] else 0.0
            done = terminated or truncated
        returns.append(info["episode_return"])
        if mission == "return":
            success = info["termination"] == "landed_on_pad"
            extra = (f"touchdown {info['pad_downrange'] / 1000:7.2f} km from pad"
                     f"{'  ON PAD' if success else ''}")
        else:
            success = info["termination"] == "time_limit" and info["orbit"]
            extra = f"e {info['eccentricity']:5.3f}{'  HELD ORBIT' if success else ''}"
        successes += success
        apo = info["apoapsis"]
        apo_s = "inf" if not math.isfinite(apo) else f"{apo / 1000:6.1f}"
        print(f"ep {ep + 1:2d}: {info['termination']:<14} return {info['episode_return']:7.1f}  "
              f"in space {space_s:5.0f} s  "
              f"final orbit {info['periapsis'] / 1000:6.1f} x {apo_s} km  "
              f"dv used {info['dv_used']:4.0f} m/s  {extra}")
    print("-" * 64)
    label = "landed on the pad" if mission == "return" else "held orbit to time limit"
    print(f"{label} {successes}/{episodes}  "
          f"mean return {sum(returns) / len(returns):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="orbit", choices=sorted(MISSIONS))
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument(
        "--device",
        default="auto",
        help="torch device for train/eval/watch (examples: auto, cpu, cuda, mps)",
    )
    ap.add_argument("--out", default=None,
                    help="model save path (default: models/ppo_orbit_<timestamp>)")
    ap.add_argument("--watch", default=None, help="path to a saved model to watch")
    ap.add_argument("--eval", type=int, default=0, help="evaluate N episodes headless")
    args = ap.parse_args()

    if args.eval:
        if not (args.watch or args.out):
            raise SystemExit("--eval needs a model: --watch models/<run>.zip")
        evaluate(args.watch or args.out, args.eval, args.device, args.mission)
        return

    if args.watch:
        import pygame

        env = MISSIONS[args.mission](render_mode="human")
        model = PPO.load(args.watch, device=args.device)
        obs, info = env.reset()
        env.render()  # create the window before polling its events
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    env.close()
                    return
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"episode over: {info['termination']}, "
                      f"orbit {info['periapsis'] / 1000:.0f} x {info['apoapsis'] / 1000:.0f} km")
                obs, info = env.reset()

    # VecMonitor inside VecNormalize so the logged ep_rew_mean stays in raw
    # (interpretable) units; only the rewards PPO sees are normalized. Obs are
    # already hand-normalized by the env.
    vec = VecNormalize(
        VecMonitor(SubprocVecEnv([MISSIONS[args.mission] for _ in range(args.envs)])),
        norm_obs=False,
        norm_reward=True,
        gamma=GAMMA,
    )
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=2048,
        batch_size=512,
        gamma=GAMMA,
        gae_lambda=0.98,
        ent_coef=0.003,
        device=args.device,
        seed=0,
        verbose=1,
    )
    out = args.out or time.strftime(f"models/ppo_{args.mission}_%Y%m%d_%H%M%S")
    model.learn(total_timesteps=args.steps)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    vec.save(out + "_vecnorm.pkl")
    print(f"saved to {out}.zip")


if __name__ == "__main__":
    main()
