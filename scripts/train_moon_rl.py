"""Train the lunar landing policy (PPO with a reachable-state curriculum).

    python scripts/train_moon_rl.py --name my-run --hypothesis "what this tries"
    python scripts/train_moon_rl.py --name polish --resume models/my-run/best.zip \\
        --lr 1e-4 --bc-coef 0.05 --std-floor 0.05 --level0 1.0
    python scripts/train_moon_rl.py --eval 32 --model models/my-run/best.zip
    python scripts/train_moon_rl.py --watch models/my-run/best.zip

Every flag is a field of `orbital_gym.rl.trainer.TrainConfig` (reward knobs
as `--r-<field>`); the defaults are the recipe behind the reference result.
Runs are logged to MLflow through `scripts/explog.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import explog

from orbital_gym.rl.landing import RewardConfig
from orbital_gym.rl.trainer import TrainConfig, TrainingRun, evaluate_checkpoint


def _add_fields(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    """One flag per dataclass field, typed from its default."""
    for f in dataclasses.fields(cls):
        if f.name == "reward":
            continue
        flag = f"--{prefix}{f.name.replace('_', '-')}"
        default: Any = f.default if f.default is not dataclasses.MISSING else None
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            parser.add_argument(flag, type=int, nargs="+", default=list(default))
        else:
            parser.add_argument(flag, type=type(default), default=default)


def parse(argv: list[str] | None = None) -> tuple[TrainConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_fields(ap, TrainConfig)
    _add_fields(ap, RewardConfig, prefix="r-")
    ap.add_argument("--eval", type=int, default=0, help="evaluate --model on this many sites")
    ap.add_argument("--model", default=None)
    ap.add_argument("--watch", default=None, help="fly this model in a window")
    args = ap.parse_args(argv)
    base = TrainConfig()
    reward = RewardConfig(
        **{f.name: getattr(args, f"r_{f.name}") for f in dataclasses.fields(RewardConfig)}
    )
    fields = {
        f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig) if f.name != "reward"
    }
    fields["net"] = tuple(fields["net"])
    # the reference reward lives in TrainConfig's default; the CLI overrides it
    # field by field, so untouched flags keep the reference values
    for f in dataclasses.fields(RewardConfig):
        if getattr(args, f"r_{f.name}") == f.default:
            setattr(reward, f.name, getattr(base.reward, f.name))
    return TrainConfig(reward=reward, **fields), args


def main() -> None:
    cfg, args = parse()
    if args.watch or args.eval:
        evaluate_checkpoint(cfg, args.model or args.watch, args.eval or 8, watch=bool(args.watch))
        return
    TrainingRun(cfg, explog).run()


if __name__ == "__main__":
    main()
