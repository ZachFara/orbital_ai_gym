"""The benchmark protocol is fixed, the baselines run on every kind of env,
and the autopilot rows are what the pilots are known to do."""

from __future__ import annotations

import orbital_gym  # noqa: F401
from orbital_gym.benchmark import ENV_IDS, protocol, run, table


def test_protocol_is_fixed_and_distinct_per_env() -> None:
    for env_id in ENV_IDS:
        eps = protocol(env_id)
        assert eps == protocol(env_id)  # deterministic
        assert len({(e.seed, e.site) for e in eps}) == len(eps)
    assert all(e.site is not None for e in protocol("ToMoon-v0"))
    assert all(e.site is None for e in protocol("OrbitAndReturn-v0"))
    assert len(protocol("MoonLandingFromOrbit-v0", 4)) == 4


def test_autopilot_rows_land_and_random_rows_do_not() -> None:
    pad = run("OrbitAndReturn-v0", "autopilot", episodes=1)
    assert pad is not None and pad.success == 1.0 and pad.terminations == {"landed_on_pad": 1}
    moon = run("MoonLandingFromOrbit-v0", "autopilot", episodes=2)
    assert moon is not None and moon.episodes == 2
    assert all(t.startswith("landed_moon") for t in moon.terminations)
    rnd = run("TwoStageToOrbit-v0", "random", episodes=2)
    assert rnd is not None and rnd.success == 0.0
    assert run("ToMoon-v0", "ppo") is None  # no model given: no row
    md = table([pad, moon, rnd])
    assert md.count("\n") == 4 and "`OrbitAndReturn-v0` | autopilot | 1 | 100%" in md
