"""Reproduce the benchmark table: every registered env under the random
policy, the scripted autopilot and (where a model is given) a trained PPO.

    python scripts/benchmark.py                                   # random + autopilot, all envs
    python scripts/benchmark.py --ppo MoonLandingFromOrbit-v0=models/ppo-zem-v15-polish/best_100pct.zip
    python scripts/benchmark.py --envs ToMoon-v0 ToMars-v0 --episodes 4
    python scripts/benchmark.py --record docs/media                # a GIF per (env, baseline)

The protocol (fixed sites and seeds per env) is `orbital_gym.benchmark`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orbital_gym  # noqa: F401  (registers the ids)
from orbital_gym.benchmark import ENV_IDS, POLICIES, Row, run, table


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--envs", nargs="+", default=list(ENV_IDS), choices=list(ENV_IDS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument(
        "--ppo", action="append", default=[], metavar="ENV_ID=PATH", help="a model per env"
    )
    ap.add_argument("--episodes", type=int, default=None, help="override the protocol's count")
    ap.add_argument(
        "--record", default=None, help="directory for a GIF of the first episode of each row"
    )
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    models = dict(item.split("=", 1) for item in args.ppo)
    rows: list[Row] = []
    for env_id in args.envs:
        for policy in args.policies:
            record = None
            if args.record and policy != "random":
                os.makedirs(args.record, exist_ok=True)
                record = os.path.join(args.record, f"{env_id.replace('-v0', '')}_{policy}.gif")
            row = run(env_id, policy, models, args.episodes, record)
            if row is None:
                continue
            rows.append(row)
            print(
                f"{env_id:<26} {policy:<9} {row.episodes:3d} ep  success {row.success:4.0%}  "
                f"return {row.ret:8.1f}  dv {row.dv_used:5.0f}  steps {row.steps:6.0f}  ({row.seconds:.0f} s)",
                flush=True,
            )
    md = table(rows)
    print()
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
