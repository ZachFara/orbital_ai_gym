"""The training configuration: its defaults are the reference recipe, the CLI
is generated from it field by field, and the evaluation summary is right."""

from __future__ import annotations

import dataclasses

import pytest
import torch as th

from orbital_gym.rl.landing import RewardConfig
from orbital_gym.rl.trainer import TrainConfig, summarize, widen_policy_params


def test_defaults_are_the_reference_recipe() -> None:
    cfg = TrainConfig()
    assert cfg.throttle_map == "exp"
    assert cfg.extra_obs
    assert cfg.gamma == pytest.approx(0.9995)
    assert cfg.gae_lambda == pytest.approx(0.98)
    assert cfg.bound_coef > 0.0 and cfg.std_floor > 0.0
    assert cfg.bc_epochs > 0 and cfg.bc_coef > 0.0
    assert cfg.reward.w_miss == pytest.approx(2.0)
    assert cfg.reward.fuel == pytest.approx(0.0005)
    assert cfg.reward.hover_cost == pytest.approx(0.02)
    assert cfg.reward.hover_alt == 500.0
    flat = cfg.flat()
    assert flat["r_w_miss"] == 2.0 and "reward" not in flat


def test_cli_flags_override_fields_and_keep_the_rest() -> None:
    from scripts.train_moon_rl import parse

    cfg, _ = parse([])
    assert cfg == TrainConfig()
    cfg, args = parse(["--gamma", "0.99", "--no-extra-obs", "--net", "64", "64", "--r-w-miss", "3"])
    assert cfg.gamma == 0.99 and not cfg.extra_obs and cfg.net == (64, 64)
    assert cfg.reward.w_miss == 3.0
    assert cfg.reward.hover_alt == TrainConfig().reward.hover_alt  # untouched: reference value
    assert args.eval == 0
    for f in dataclasses.fields(RewardConfig):  # every reward knob has a flag
        assert hasattr(args, f"r_{f.name}")


def test_summarize_reads_the_episode_rows() -> None:
    rows = [
        dict(
            on_site=1,
            soft=1,
            termination="landed_moon_site",
            miss_km=0.5,
            rel_speed=4.0,
            dv_spent=1800.0,
            native_return=500.0,
        ),
        dict(
            on_site=0,
            soft=0,
            termination="crashed",
            miss_km=30.0,
            rel_speed=90.0,
            dv_spent=900.0,
            native_return=-100.0,
        ),
    ]
    out = summarize(rows, prefix_return=2500.0)
    assert out["on_site"] == 0.5 and out["crash"] == 0.5 and out["timeout"] == 0.0
    assert out["full_mission"] == pytest.approx(2700.0)


def test_widening_inserts_zero_input_columns_and_keeps_the_rest() -> None:
    g = th.Generator().manual_seed(0)
    old = {
        "mlp_extractor.policy_net.0.weight": th.randn(4, 6, generator=g),
        "mlp_extractor.value_net.0.weight": th.randn(4, 6, generator=g),
        "action_net.weight": th.randn(3, 4, generator=g),
    }
    new = widen_policy_params(old, at=4, count=2)
    for key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        w = new[key]
        assert w.shape == (4, 8)
        assert th.equal(w[:, :4], old[key][:, :4]) and th.equal(w[:, 6:], old[key][:, 4:])
        assert th.all(w[:, 4:6] == 0.0)
    assert new["action_net.weight"] is old["action_net.weight"]
    # a padded observation gives the same pre-activation as the old one
    x_old = th.randn(5, 6, generator=g)
    x_new = th.cat([x_old[:, :4], th.randn(5, 2, generator=g), x_old[:, 4:]], dim=1)
    assert th.allclose(
        x_new @ new["mlp_extractor.policy_net.0.weight"].T,
        x_old @ old["mlp_extractor.policy_net.0.weight"].T,
    )
