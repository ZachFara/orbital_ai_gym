"""Train PPO on TwoStageToOrbit with Stable-Baselines3.

    pip install -e ".[train]"
    python scripts/train_ppo.py --steps 3_000_000

Uses terminate_on_orbit=True: reaching a stable orbit immediately banks the
duration reward the agent would have earned by coasting to the time limit,
which is reward-equivalent but makes episodes (and training) much shorter.
Watch a trained model with --watch models/ppo_orbit.zip.
"""

from __future__ import annotations

import argparse

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
except ImportError:
    raise SystemExit("stable-baselines3 not installed; run: pip install -e '.[train]'")

from orbital_gym.two_stage_env import TwoStageToOrbitEnv


def make_env():
    return TwoStageToOrbitEnv(terminate_on_orbit=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--out", default="models/ppo_orbit")
    ap.add_argument("--watch", default=None, help="path to a saved model to watch")
    args = ap.parse_args()

    if args.watch:
        env = TwoStageToOrbitEnv(render_mode="human")
        model = PPO.load(args.watch)
        obs, info = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"episode over: {info['termination']}, "
                      f"orbit {info['periapsis'] / 1000:.0f} x {info['apoapsis'] / 1000:.0f} km")
                obs, info = env.reset()

    vec = VecMonitor(SubprocVecEnv([make_env for _ in range(args.envs)]))
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=2048,
        batch_size=512,
        gamma=0.999,        # long horizon: duration reward arrives late
        gae_lambda=0.98,
        ent_coef=0.003,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(args.out)
    print(f"saved to {args.out}.zip")


if __name__ == "__main__":
    main()
