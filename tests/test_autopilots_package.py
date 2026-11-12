"""The pilots live in the package; the demo script only re-exports them."""

from orbital_gym import autopilots
from orbital_gym.missions.to_orbit import TwoStageToOrbitEnv


def test_package_exports_every_pilot() -> None:
    for name in autopilots.__all__:
        assert hasattr(autopilots, name)
    env = TwoStageToOrbitEnv()
    env.reset()
    assert autopilots.Autopilot(env).mode == "ascend"


def test_demo_script_reexports_the_pilots() -> None:
    from scripts import autopilot_demo

    assert autopilot_demo.MoonAutopilot is autopilots.MoonAutopilot
    assert autopilot_demo.MarsAutopilot is autopilots.MarsAutopilot
