"""Celestial bodies on rails: the one abstraction for moons, planets and
stars that a mission adds on top of the launch planet.

A `CelestialBody` knows where it is and how fast it moves at any time (its
`CircularRails`, optionally expressed relative to the rails of the frame the
simulation runs in), how hard it pulls (with the tidal correction a
free-falling frame needs), whether it has an atmosphere, how big its sphere
of influence is, and the two instruments a lander needs about it: orbital
elements in its frame and the point where a coasting trajectory would hit
its surface. Missions own the rocket and the reward; bodies own themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import TWO_PI, elements, impact_point
from .presets import Moon, OrbitingBody

Vec = tuple[float, float]


@dataclass(frozen=True)
class CircularRails:
    """Uniform circular motion about an origin: angle(t) = phase0 + 2*pi*t/period."""

    radius: float
    period: float
    phase0: float = 0.0

    @property
    def rate(self) -> float:
        return TWO_PI / self.period

    def angle(self, t: float) -> float:
        return self.phase0 + self.rate * t

    def position(self, t: float) -> Vec:
        a = self.angle(t)
        return self.radius * math.cos(a), self.radius * math.sin(a)

    def velocity(self, t: float) -> Vec:
        a = self.angle(t)
        s = self.rate * self.radius
        return -s * math.sin(a), s * math.cos(a)


@dataclass(frozen=True)
class Atmosphere:
    rho0: float
    scale_height: float
    space_altitude: float

    def density(self, altitude: float) -> float:
        if self.rho0 <= 0.0 or altitude >= self.space_altitude:
            return 0.0
        return self.rho0 * math.exp(-max(0.0, altitude) / self.scale_height)


class CelestialBody:
    """A body in the simulation frame.

    `rails` is the body's circular motion about its primary; `frame` is the
    simulation frame's own motion about that same primary (None when the
    frame IS the primary, as the launch planet is for its moon). The body's
    position in the frame is rails minus frame. A star at the primary has
    no rails of its own.

    `tidal=True` subtracts the body's pull on the frame origin from its pull
    on the rocket, which is what a frame fixed to a free-falling planet
    requires. ToMoon keeps `tidal=False` (its planet is pinned at the
    origin and its published scores depend on it).
    """

    def __init__(
        self,
        name: str,
        kind: str,
        radius: float,
        mu: float,
        soi: float,
        rails: CircularRails | None = None,
        frame: CircularRails | None = None,
        atmosphere: Atmosphere | None = None,
        tidal: bool = False,
        soft_radius_fraction: float = 0.25,
    ):
        self.name = name
        self.kind = kind  # "moon" | "planet" | "star": the renderer's style
        self.radius = radius
        self.mu = mu
        self.soi = soi
        self.rails = rails
        self.frame = frame
        self.atmosphere = atmosphere
        self.tidal = tidal
        self.soft2 = (soft_radius_fraction * radius) ** 2

    @classmethod
    def from_preset(
        cls,
        name: str,
        kind: str,
        preset: OrbitingBody | Moon,
        primary_mu: float,
        frame: CircularRails | None = None,
        phase0: float = 0.0,
        tidal: bool = False,
    ) -> CelestialBody:
        soi = preset.orbit_radius * (preset.mu / primary_mu) ** 0.4
        atmo = (
            Atmosphere(preset.rho0, preset.scale_height, preset.space_altitude)
            if isinstance(preset, OrbitingBody) and preset.rho0 > 0.0
            else None
        )
        return cls(
            name,
            kind,
            preset.radius,
            preset.mu,
            soi,
            CircularRails(preset.orbit_radius, preset.orbit_period, phase0),
            frame,
            atmo,
            tidal,
        )

    # -- where it is ------------------------------------------------------------

    def angle(self, t: float) -> float:
        """The body's angle about its primary (its own rotation, too: every
        body here is tidally locked, so a surface site keeps this angle plus
        its offset)."""
        return self.rails.angle(t) if self.rails is not None else 0.0

    def position(self, t: float) -> Vec:
        x, y = self.rails.position(t) if self.rails is not None else (0.0, 0.0)
        if self.frame is not None:
            fx, fy = self.frame.position(t)
            x, y = x - fx, y - fy
        return x, y

    def velocity(self, t: float) -> Vec:
        vx, vy = self.rails.velocity(t) if self.rails is not None else (0.0, 0.0)
        if self.frame is not None:
            fvx, fvy = self.frame.velocity(t)
            vx, vy = vx - fvx, vy - fvy
        return vx, vy

    def site_world(self, t: float, site_angle: float) -> Vec:
        bx, by = self.position(t)
        a = self.angle(t) + site_angle
        return bx + self.radius * math.cos(a), by + self.radius * math.sin(a)

    # -- the rocket relative to it ---------------------------------------------------

    def relative(
        self, rx: float, ry: float, vx: float, vy: float, t: float
    ) -> tuple[float, float, float, float]:
        bx, by = self.position(t)
        bvx, bvy = self.velocity(t)
        return rx - bx, ry - by, vx - bvx, vy - bvy

    def distance(self, rx: float, ry: float, t: float) -> float:
        bx, by = self.position(t)
        return math.hypot(rx - bx, ry - by)

    def altitude(self, rx: float, ry: float, t: float) -> float:
        return self.distance(rx, ry, t) - self.radius

    def in_soi(self, rx: float, ry: float, t: float) -> bool:
        return self.distance(rx, ry, t) <= self.soi

    def relative_speed(self, rx: float, ry: float, vx: float, vy: float, t: float) -> float:
        _, _, wx, wy = self.relative(rx, ry, vx, vy, t)
        return math.hypot(wx, wy)

    def radial_tangential(
        self, rx: float, ry: float, vx: float, vy: float, t: float
    ) -> tuple[float, float]:
        """Body-relative velocity split into (radial, tangential) in the
        rocket's local frame about this body, with the mission envs'
        convention: radial is up, tangential is clockwise-positive."""
        px, py, wx, wy = self.relative(rx, ry, vx, vy, t)
        d = max(1.0, math.hypot(px, py))
        ux, uy = px / d, py / d
        return wx * ux + wy * uy, wx * uy - wy * ux

    def elements(
        self, rx: float, ry: float, vx: float, vy: float, t: float
    ) -> tuple[float, float, float, float]:
        """(periapsis_alt, apoapsis_alt, energy, ecc) in the body's frame."""
        px, py, wx, wy = self.relative(rx, ry, vx, vy, t)
        return elements(self.mu, self.radius, px, py, wx, wy)

    def impact(
        self, rx: float, ry: float, vx: float, vy: float, t: float, site_angle: float
    ) -> tuple[bool, float]:
        """(impacting, signed angle from the site to the predicted touchdown)."""
        px, py, wx, wy = self.relative(rx, ry, vx, vy, t)
        return impact_point(self.mu, self.radius, px, py, wx, wy, self.angle(t) + site_angle)

    # -- forces ----------------------------------------------------------------------

    def gravity(self, rx: float, ry: float, t: float) -> Vec:
        """Acceleration of the rocket toward the body, softened near its
        centre; minus the body's pull on the frame origin when `tidal`."""
        bx, by = self.position(t)
        dx, dy = bx - rx, by - ry
        r2 = max(dx * dx + dy * dy, self.soft2)
        inv = self.mu / (r2 * math.sqrt(r2))
        ax, ay = dx * inv, dy * inv
        if self.tidal:
            b2 = bx * bx + by * by
            invb = self.mu / (b2 * math.sqrt(b2))
            ax -= bx * invb
            ay -= by * invb
        return ax, ay

    def drag(
        self, rx: float, ry: float, vx: float, vy: float, t: float, cd_area: float, mass: float
    ) -> Vec:
        """Aerodynamic deceleration in the body's atmosphere, on the
        body-relative velocity; zero for an airless body or above its top."""
        if self.atmosphere is None:
            return 0.0, 0.0
        px, py, wx, wy = self.relative(rx, ry, vx, vy, t)
        alt = math.hypot(px, py) - self.radius
        rho = self.atmosphere.density(alt)
        if rho <= 0.0:
            return 0.0, 0.0
        speed = math.hypot(wx, wy)
        k = -0.5 * rho * cd_area * speed / mass
        return k * wx, k * wy

    def density(self, altitude: float) -> float:
        return self.atmosphere.density(altitude) if self.atmosphere is not None else 0.0
