"""TwoStageToOrbit: a 2D two-stage-rocket-to-orbit Gymnasium environment.

The world is a planet-centered 2D plane. You launch from the surface at
position (0, R) with the rocket pointing straight up. Below the atmosphere
cutoff the reward is altitude gained; above it the reward is time spent up
there. Since fuel is finite, the only way to keep collecting the duration
reward is to be moving sideways fast enough to orbit -- the reward function
teaches the gravity turn by itself.

Action space (Box, [-1, 1]):
    a[0]  throttle command, mapped to [0, 1]
    a[1]  attitude turn-rate command, mapped to [-max_turn_rate, +max_turn_rate]
    a[2]  (only when auto_stage=False) fire staging when > 0

Observation (Box(9,), float32), all normalized:
    0  altitude / space_altitude
    1  radial (vertical) velocity / v_ref
    2  tangential (horizontal) velocity / v_ref, positive = "east" (+x at pad)
    3  sin(pitch)   pitch = attitude angle above the local horizon
    4  cos(pitch)
    5  propellant fraction remaining in the current stage
    6  current stage index / (num_stages - 1)
    7  current mass / launch mass
    8  fraction of the episode time limit elapsed

where v_ref is the circular orbital velocity just above the atmosphere.

Reward:
    climb  potential-based: climb_reward * d(min(alt, cutoff)) / cutoff.
           Sums to exactly `climb_reward` for reaching the cutoff, refunds on
           the way back down, so it cannot be farmed by hopping.
    space  space_rate per second while altitude > cutoff.
    orbit  orbit_rate per second while in a stable orbit (periapsis above the
           cutoff), scaled by circularity: max(0, 1 - ecc / orbit_ecc_scale).
           Without this term a barely-bound ellipse flung out to a huge
           apoapsis collects the space reward exactly as well as a tidy
           circular orbit; with it, circularizing roughly doubles the pay.
    turn   -turn_penalty per second, scaled by |turn command|. A whisper of
           control cost, applied in step() for every mission, so policies
           don't tumble end over end whenever attitude doesn't matter.
    crash  -crash_penalty on impact above crash_speed.

Termination: crash, gentle landing after flight, guaranteed escape
(hyperbolic trajectory above the atmosphere with no propellant left -- this
closes the "burn straight up and cruise to infinity" exploit), and optionally
stable orbit (periapsis above the atmosphere) with a bonus equal to the
duration-plus-orbit reward the agent would have collected by coasting to the
time limit at its current circularity.
Truncation at max_episode_steps (handled internally).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..presets import G0, PLANETS, ROCKETS, Planet, Rocket, total_delta_v

TWO_PI = 2.0 * math.pi


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


def _elements(mu: float, body_radius: float, rx: float, ry: float,
              vx: float, vy: float):
    """Orbital elements of a body-relative state:
    (periapsis_alt, apoapsis_alt, specific_energy, eccentricity).

    Apoapsis is +inf on escape trajectories.
    """
    rn = math.hypot(rx, ry)
    v2 = vx * vx + vy * vy
    energy = 0.5 * v2 - mu / rn
    h = rx * vy - ry * vx
    ecc2 = 1.0 + 2.0 * energy * h * h / (mu * mu)
    ecc = math.sqrt(max(0.0, ecc2))
    if energy >= 0.0:
        peri = (h * h / mu) / (1.0 + ecc) - body_radius if ecc > 0 else rn - body_radius
        return peri, math.inf, energy, ecc
    sma = -mu / (2.0 * energy)
    return sma * (1.0 - ecc) - body_radius, sma * (1.0 + ecc) - body_radius, energy, ecc


class TwoStageToOrbitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    OBS_SIZE = 9  # subclasses that append observations override this

    def __init__(
        self,
        planet: str | Planet = "easy",
        rocket: str | Rocket | None = None,
        auto_stage: bool = True,
        step_duration: float = 1.0,
        physics_dt: float = 0.1,
        max_episode_steps: int = 4000,
        climb_reward: float = 100.0,
        space_rate: float = 1.0,
        orbit_rate: float = 1.0,
        orbit_ecc_scale: float = 0.25,
        turn_penalty: float = 0.01,
        crash_penalty: float = 100.0,
        crash_speed: float = 5.0,
        terminate_on_orbit: bool = False,
        render_mode: str | None = None,
    ):
        self.planet = PLANETS[planet] if isinstance(planet, str) else planet
        if rocket is None:
            rocket = ROCKETS.get(self.planet.name, ROCKETS["easy"])
        self.rocket = ROCKETS[rocket] if isinstance(rocket, str) else rocket

        self.auto_stage = auto_stage
        self.step_duration = step_duration
        self.physics_dt = physics_dt
        self._substeps = max(1, round(step_duration / physics_dt))
        self.max_episode_steps = max_episode_steps
        self.climb_reward = climb_reward
        self.space_rate = space_rate
        self.orbit_rate = orbit_rate
        self.orbit_ecc_scale = orbit_ecc_scale
        self.turn_penalty = turn_penalty
        self.crash_penalty = crash_penalty
        self.crash_speed = crash_speed
        self.terminate_on_orbit = terminate_on_orbit

        self.v_ref = self.planet.circular_velocity(self.planet.space_altitude)

        n_act = 2 if auto_stage else 3
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-20.0, 20.0, shape=(self.OBS_SIZE,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = None

        self.reset(seed=None)

    # --- Gym API -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        p = self.planet
        self._rx, self._ry = 0.0, p.radius
        self._vx, self._vy = 0.0, 0.0
        self._theta = math.pi / 2.0  # pointing straight up (local vertical)
        self._stage = 0
        self._fuel = self.rocket.stages[0].prop_mass
        self._mass = self.rocket.launch_mass
        self._dv_used = 0.0
        self._steps = 0
        self._time = 0.0
        self._throttle_actual = 0.0
        self._airborne = False
        self._terminal: str | None = None
        self._episode_return = 0.0
        self._debris: dict | None = None
        self._trail: list[tuple[float, float]] = [(self._rx, self._ry)]
        return self._obs(), self._info()

    def step(self, action):
        assert self._terminal is None, "step() called on a finished episode; call reset()"
        a = np.asarray(action, dtype=np.float64).flatten()
        throttle = float(np.clip((a[0] + 1.0) * 0.5, 0.0, 1.0))
        turn_cmd = float(np.clip(a[1], -1.0, 1.0))
        turn_rate = turn_cmd * self.rocket.max_turn_rate
        if not self.auto_stage and len(a) > 2 and a[2] > 0.0:
            self._next_stage()

        alt_before = self._altitude()
        space_time = 0.0
        for _ in range(self._substeps):
            self._substep(throttle, turn_rate)
            if self._altitude() > self.planet.space_altitude:
                space_time += self.physics_dt
            if self._terminal is not None:
                break

        self._steps += 1
        self._trail.append((self._rx, self._ry))
        if len(self._trail) > 6000:
            del self._trail[:2000]
        alt_after = self._altitude()
        cutoff = self.planet.space_altitude
        reward = self._step_reward(alt_before, alt_after, space_time)
        reward -= self.turn_penalty * abs(turn_cmd) * self.step_duration

        if self._terminal == "crashed":
            reward -= self.crash_penalty

        peri, apo, energy, ecc = self._orbit_elements()
        if self._terminal is None:
            if (
                alt_after > cutoff
                and self._propellant_left() <= 0.0
                and self._is_escaping()
            ):
                self._terminal = "escaped"
            elif (
                self.terminate_on_orbit
                and alt_after > cutoff
                and energy < 0.0
                and peri > cutoff
            ):
                remaining = max(0, self.max_episode_steps - self._steps)
                circ = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
                reward += (
                    (self.space_rate + self.orbit_rate * circ)
                    * self.step_duration * remaining
                )
                self._terminal = "orbit"

        terminated = self._terminal is not None
        truncated = not terminated and self._steps >= self.max_episode_steps
        if truncated:
            self._terminal = "time_limit"

        self._episode_return += reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from ..rendering import Renderer
            self._renderer = Renderer(self, self.render_mode)
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- Physics ------------------------------------------------------------

    def _extra_acceleration(self, rx: float, ry: float, vx: float, vy: float) -> tuple[float, float]:
        """Mission-specific acceleration term (default none).

        Subclasses can override this to add third-body gravity, SRP, etc.
        Returned values are added to translational acceleration in m/s^2.
        """
        return 0.0, 0.0

    def _substep(self, throttle: float, turn_rate: float) -> None:
        dt = self.physics_dt
        p = self.planet
        self._theta = self._theta + turn_rate * dt

        rx, ry, vx, vy = self._rx, self._ry, self._vx, self._vy
        rn = math.hypot(rx, ry)
        alt = rn - p.radius

        g = -p.mu / (rn * rn * rn)
        ax, ay = g * rx, g * ry
        ax_extra, ay_extra = self._extra_acceleration(rx, ry, vx, vy)
        ax += ax_extra
        ay += ay_extra

        stage = self.rocket.stages[self._stage]
        self._throttle_actual = 0.0
        if throttle > 0.0 and self._fuel > 0.0:
            force = throttle * stage.thrust
            mdot = force / (stage.isp * G0)
            burn = min(self._fuel, mdot * dt)
            force *= burn / (mdot * dt)
            m0 = self._mass
            self._fuel -= burn
            self._mass -= burn
            self._dv_used += stage.isp * G0 * math.log(m0 / self._mass)
            acc = force / self._mass
            ax += acc * math.cos(self._theta)
            ay += acc * math.sin(self._theta)
            self._throttle_actual = force / stage.thrust

        rho = p.density(alt)
        if rho > 0.0:
            speed = math.hypot(vx, vy)
            k = -0.5 * rho * stage.cd_area * speed / self._mass
            ax += k * vx
            ay += k * vy

        vx += ax * dt
        vy += ay * dt
        rx += vx * dt
        ry += vy * dt

        rn = math.hypot(rx, ry)
        if rn <= p.radius:
            ux, uy = rx / rn, ry / rn
            impact_speed = -(vx * ux + vy * uy)
            rx, ry = ux * p.radius, uy * p.radius
            vx = vy = 0.0
            if impact_speed > self.crash_speed:
                self._terminal = "crashed"
            elif self._airborne:
                self._terminal = "landed"

        self._rx, self._ry, self._vx, self._vy = rx, ry, vx, vy
        self._time += dt
        if rn - p.radius > 100.0:
            self._airborne = True

        d = self._debris
        if d is not None:  # the jettisoned stage falls ballistically (visual only)
            dn = math.hypot(d["rx"], d["ry"])
            gd = -p.mu / (dn * dn * dn)
            ax_d, ay_d = gd * d["rx"], gd * d["ry"]
            rho_d = p.density(dn - p.radius)
            if rho_d > 0.0:
                sp = math.hypot(d["vx"], d["vy"])
                kd = -0.5 * rho_d * d["cd_area"] * sp / d["mass"]
                ax_d += kd * d["vx"]
                ay_d += kd * d["vy"]
            d["vx"] += ax_d * dt
            d["vy"] += ay_d * dt
            d["rx"] += d["vx"] * dt
            d["ry"] += d["vy"] * dt
            d["theta"] += d["spin"] * dt
            d["age"] += dt
            if math.hypot(d["rx"], d["ry"]) <= p.radius or d["age"] > 300.0:
                self._debris = None

        if self.auto_stage and self._fuel <= 0.0 and self._stage < len(self.rocket.stages) - 1:
            self._next_stage()

    def _next_stage(self) -> None:
        if self._stage >= len(self.rocket.stages) - 1:
            return
        dropped = self.rocket.stages[self._stage]
        self._debris = {
            "rx": self._rx, "ry": self._ry,
            "vx": self._vx - 3.0 * math.cos(self._theta),
            "vy": self._vy - 3.0 * math.sin(self._theta),
            "theta": self._theta, "spin": 0.5 if self._steps % 2 else -0.5,
            "age": 0.0, "cd_area": dropped.cd_area,
            "mass": max(1.0, dropped.dry_mass),
        }
        self._mass -= dropped.dry_mass + self._fuel
        self._stage += 1
        self._fuel = self.rocket.stages[self._stage].prop_mass

    # --- Reward ---------------------------------------------------------------

    def _climb_term(self, alt_before: float, alt_after: float) -> float:
        """Potential-based climb reward; sums to climb_reward for reaching space."""
        cutoff = self.planet.space_altitude
        return (
            self.climb_reward
            * (min(alt_after, cutoff) - min(alt_before, cutoff))
            / cutoff
        )

    def _step_reward(self, alt_before: float, alt_after: float, space_time: float) -> float:
        """Per-step reward, called after the substeps (terminal already set if
        the rocket landed or crashed this step). Missions override this; the
        turn penalty is applied on top by step(), for every mission."""
        reward = self._climb_term(alt_before, alt_after) + self.space_rate * space_time
        if space_time > 0.0 and self.orbit_rate > 0.0:
            peri, _, energy, ecc = self._orbit_elements()
            if energy < 0.0 and peri > self.planet.space_altitude:
                circ = max(0.0, 1.0 - ecc / self.orbit_ecc_scale)
                reward += self.orbit_rate * circ * space_time
        return reward

    # --- Derived quantities ---------------------------------------------------

    def _altitude(self) -> float:
        return math.hypot(self._rx, self._ry) - self.planet.radius

    def _local_frame(self):
        """Unit vectors: radial (up) and tangential ("east", +x at the pad)."""
        rn = math.hypot(self._rx, self._ry)
        ux, uy = self._rx / rn, self._ry / rn
        return ux, uy, uy, -ux

    def _pitch(self) -> float:
        """Attitude angle above the local horizon, in (-pi, pi]."""
        ux, uy, tx, ty = self._local_frame()
        return _wrap_pi(self._theta - math.atan2(ty, tx))

    def _orbit_elements(self):
        """Returns (periapsis_alt, apoapsis_alt, specific_energy, eccentricity)
        relative to the planet. Apoapsis is +inf on escape trajectories.
        """
        p = self.planet
        return _elements(p.mu, p.radius, self._rx, self._ry, self._vx, self._vy)

    def _is_escaping(self) -> bool:
        """Guaranteed to leave: hyperbolic relative to the planet. Missions
        with more bodies override this -- near a moon the planet-centric
        elements say nothing about where the rocket is actually headed."""
        return self._orbit_elements()[2] >= 0.0

    def _propellant_left(self) -> float:
        upper = sum(s.prop_mass for s in self.rocket.stages[self._stage + 1:])
        return self._fuel + upper

    def delta_v_remaining(self) -> float:
        """Ideal delta-v left, from the current mass and remaining propellant."""
        dv = 0.0
        m = self._mass
        fuel = self._fuel
        for i in range(self._stage, len(self.rocket.stages)):
            s = self.rocket.stages[i]
            dv += s.isp * G0 * math.log(m / (m - fuel))
            m -= fuel + s.dry_mass
            if i + 1 < len(self.rocket.stages):
                fuel = self.rocket.stages[i + 1].prop_mass
        return dv

    def total_delta_v(self) -> float:
        """Ideal delta-v of the full rocket at launch (Tsiolkovsky)."""
        return total_delta_v(self.rocket)

    def _obs(self) -> np.ndarray:
        ux, uy, tx, ty = self._local_frame()
        vr = self._vx * ux + self._vy * uy
        vt = self._vx * tx + self._vy * ty
        pitch = self._pitch()
        stage_frac = self._stage / max(1, len(self.rocket.stages) - 1)
        obs = np.array(
            [
                self._altitude() / self.planet.space_altitude,
                vr / self.v_ref,
                vt / self.v_ref,
                math.sin(pitch),
                math.cos(pitch),
                self._fuel / self.rocket.stages[self._stage].prop_mass,
                stage_frac,
                self._mass / self.rocket.launch_mass,
                self._steps / self.max_episode_steps,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -20.0, 20.0)

    def _info(self) -> dict[str, Any]:
        ux, uy, tx, ty = self._local_frame()
        vr = self._vx * ux + self._vy * uy
        vt = self._vx * tx + self._vy * ty
        peri, apo, energy, ecc = self._orbit_elements()
        alt = self._altitude()
        rho = self.planet.density(alt)
        speed = math.hypot(self._vx, self._vy)
        return {
            "time": self._time,
            "altitude": alt,
            "speed": speed,
            "vr": vr,
            "vt": vt,
            "pitch_deg": math.degrees(self._pitch()),
            "mass": self._mass,
            "stage": self._stage,
            "fuel": self._fuel,
            "periapsis": peri,
            "apoapsis": apo,
            "eccentricity": ecc,
            "circularity": max(0.0, 1.0 - ecc / self.orbit_ecc_scale),
            "energy": energy,
            "dynamic_pressure": 0.5 * rho * speed * speed,
            "dv_used": self._dv_used,
            "dv_remaining": self.delta_v_remaining(),
            "episode_return": self._episode_return,
            "in_space": alt > self.planet.space_altitude,
            "orbit": bool(
                alt > self.planet.space_altitude
                and energy < 0.0
                and peri > self.planet.space_altitude
            ),
            "termination": self._terminal,
        }

    # Test/debug hook: place the rocket anywhere with a given velocity.
    def _teleport(self, rx, ry, vx, vy, theta=None):
        self._rx, self._ry, self._vx, self._vy = rx, ry, vx, vy
        if theta is not None:
            self._theta = theta
        self._airborne = self._altitude() > 100.0
