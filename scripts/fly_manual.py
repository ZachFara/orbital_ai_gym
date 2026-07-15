"""Fly the rocket yourself with the keyboard.

    python scripts/fly_manual.py [--mission orbit|return|moon] [--planet easy|earth] [--auto-stage]

Controls:
    left/right   rotate
    up/down      throttle +/- 10%
    space        toggle full throttle / cutoff
    s            fire next stage (manual staging is the default here)
    1 / 2 / 3    time warp x1 / x5 / x20
    r            reset    esc quits

Each sim step is 1 s, rendered at 30 fps, so x1 warp is ~30x real time.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from orbital_gym.missions.orbit_and_return import OrbitAndReturnEnv
from orbital_gym.missions.to_moon import ToMoonEnv
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="orbit", choices=["orbit", "return", "moon"])
    ap.add_argument("--planet", default="easy", choices=["easy", "earth"])
    ap.add_argument("--auto-stage", action="store_true")
    args = ap.parse_args()

    env_cls = {
        "orbit": TwoStageToOrbitEnv,
        "return": OrbitAndReturnEnv,
        "moon": ToMoonEnv,
    }[args.mission]
    env = env_cls(
        planet=args.planet,
        auto_stage=args.auto_stage,
        render_mode="human",
        max_episode_steps=100_000,
    )
    env.reset()
    env.render()

    throttle, warp, done = 0.0, 1, False
    while True:
        stage_cmd = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
                if event.key == pygame.K_r:
                    env.reset()
                    throttle, done = 0.0, False
                if event.key == pygame.K_SPACE:
                    throttle = 0.0 if throttle > 0.0 else 1.0
                if event.key == pygame.K_s:
                    stage_cmd = True
                if event.key == pygame.K_1:
                    warp = 1
                if event.key == pygame.K_2:
                    warp = 5
                if event.key == pygame.K_3:
                    warp = 20

        keys = pygame.key.get_pressed()
        turn = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])
        if keys[pygame.K_UP]:
            throttle = min(1.0, throttle + 0.1)
        if keys[pygame.K_DOWN]:
            throttle = max(0.0, throttle - 0.1)

        if not done:
            action = [2.0 * throttle - 1.0, turn]
            if not env.auto_stage:
                action.append(1.0 if stage_cmd else -1.0)
            for _ in range(warp):
                # render only on the last warp substep (env renders in step())
                env.render_mode = "human" if _ == warp - 1 else None
                _obs, _r, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
                env.render_mode = "human"
                action = [2.0 * throttle - 1.0, turn] + ([-1.0] if not env.auto_stage else [])
                if terminated or truncated:
                    done = True
                    print(f"episode over: {info['termination']} (press R to reset)")
                    break
        else:
            env.render()


if __name__ == "__main__":
    main()
