"""The pilot base class: what every scripted pilot shares.

A pilot is a mode machine over the env's `info` dict that returns actions
in the env's action space. The base owns the state it reads each step
(`LocalState`), the two ways of commanding the rocket (a world-frame
attitude with a throttle, or an elevation above the local horizon), and
the throttle mapping that hides `hover_control`. Mission pilots own their
modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..missions.to_orbit import TwoStageToOrbitEnv, _wrap_pi


def throttle_action(env: TwoStageToOrbitEnv, throttle: float) -> float:
    """Map a desired throttle to the a[0] the env expects. With hover_control
    the action is hover-relative (throttle = hover * exp(a[0]*K)), so invert
    that; otherwise it is the plain [0,1] -> [-1,1]. Shared by every autopilot."""
    if getattr(env, "hover_control", False):
        hover = max(1e-4, env.hover_throttle())
        a0 = math.log(max(1e-4, throttle) / hover) / env.HOVER_K
        return float(np.clip(a0, -1.0, 1.0))
    return 2.0 * throttle - 1.0


@dataclass(frozen=True)
class LocalState:
    """The rocket's state in the launch planet's local frame, as every
    mission pilot reads it at the top of a step."""

    r: np.ndarray  # position, world frame (m)
    v: np.ndarray  # velocity, world frame (m/s)
    rn: float  # distance from the planet centre
    u: np.ndarray  # radial unit vector (up)
    tw: np.ndarray  # counter-clockwise tangent: the sense the pilots orbit in
    vr: float  # radial speed
    vtw: float  # tangential speed along tw
    alt: float  # altitude above the planet
    cutoff: float  # top of the atmosphere
    apo: float  # planet-centric apoapsis altitude from info (inf when open)
    peri: float  # planet-centric periapsis altitude from info
    prograde: float  # elevation of the velocity vector above the local horizon


class Pilot:
    """Base class for the scripted pilots."""

    def __init__(self, env: TwoStageToOrbitEnv):
        self.env = env
        self.mode = "ascend"

    def __call__(self, info: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    # -- reading the state ----------------------------------------------------------

    def _local(self, info: dict[str, Any]) -> LocalState:
        e, p = self.env, self.env.planet
        r = np.array([e._rx, e._ry])
        v = np.array([e._vx, e._vy])
        rn = float(np.linalg.norm(r))
        u = r / rn
        tw = np.array([-u[1], u[0]])
        vr = float(v @ u)
        vtw = float(v @ tw)
        return LocalState(
            r=r,
            v=v,
            rn=rn,
            u=u,
            tw=tw,
            vr=vr,
            vtw=vtw,
            alt=rn - p.radius,
            cutoff=p.space_altitude,
            apo=info["apoapsis"],
            peri=info["periapsis"],
            prograde=math.atan2(vr, max(1.0, vtw)),
        )

    def _accel_avail(self, info: dict[str, Any]) -> float:
        """Thrust acceleration available at the mass reported in info."""
        return self.env.rocket.stages[self.env._stage].thrust / info["mass"]

    def _accel_now(self) -> float:
        """Thrust acceleration available at the env's current mass."""
        return self.env.rocket.stages[self.env._stage].thrust / self.env._mass

    # -- commanding the rocket ---------------------------------------------------------

    def _act_world(self, theta: float, throttle: float, gate: float | None = None) -> np.ndarray:
        """Point the thrust at world angle `theta` with `throttle`; with a
        `gate` (rad), hold the throttle at zero while the attitude error is
        larger than that, so the burn starts aligned."""
        e = self.env
        err = _wrap_pi(theta - e._theta)
        if gate is not None and abs(err) > gate:
            throttle = 0.0
        if e.attitude_control:
            ux, uy, tx, ty = e._local_frame()
            wx, wy = math.cos(theta), math.sin(theta)
            return np.array(
                [throttle_action(e, throttle), wx * ux + wy * uy, wx * tx + wy * ty],
                dtype=np.float32,
            )
        turn = np.clip(err / (e.rocket.max_turn_rate * e.step_duration), -1.0, 1.0)
        return np.array([throttle_action(e, throttle), turn], dtype=np.float32)

    def _act_elev(
        self,
        elev: float,
        throttle: float,
        u: np.ndarray,
        tw: np.ndarray,
        gate: float | None = None,
    ) -> np.ndarray:
        """Point the thrust `elev` radians above the horizon along `tw`."""
        d = math.cos(elev) * tw + math.sin(elev) * u
        return self._act_world(math.atan2(d[1], d[0]), throttle, gate)
