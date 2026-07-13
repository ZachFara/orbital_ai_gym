"""Planet and rocket presets, plus the delta-v bookkeeping helpers.

All units are SI (meters, kilograms, seconds, newtons). Isp is in seconds and
always uses standard g0 = 9.80665 regardless of the planet, per convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665  # standard gravity for Isp conversion, m/s^2


@dataclass(frozen=True)
class Planet:
    name: str
    radius: float           # m
    mu: float               # gravitational parameter GM, m^3/s^2
    rho0: float             # sea-level atmospheric density, kg/m^3
    scale_height: float     # exponential atmosphere scale height, m
    space_altitude: float   # top of atmosphere = reward cutoff altitude, m

    @property
    def surface_gravity(self) -> float:
        return self.mu / self.radius**2

    def circular_velocity(self, altitude: float) -> float:
        return math.sqrt(self.mu / (self.radius + altitude))

    def density(self, altitude: float) -> float:
        if altitude >= self.space_altitude or altitude < 0.0:
            return 0.0 if altitude >= self.space_altitude else self.rho0
        return self.rho0 * math.exp(-altitude / self.scale_height)


@dataclass(frozen=True)
class Stage:
    dry_mass: float   # kg, structure + engine (top stage: includes payload)
    prop_mass: float  # kg
    thrust: float     # N, max vacuum thrust (no pressure dependence modeled)
    isp: float        # s
    cd_area: float    # drag coefficient * reference area, m^2


@dataclass(frozen=True)
class Rocket:
    name: str
    stages: tuple[Stage, ...]  # bottom to top
    max_turn_rate: float       # rad/s attitude slew limit

    @property
    def launch_mass(self) -> float:
        return sum(s.dry_mass + s.prop_mass for s in self.stages)


def stage_delta_v(rocket: Rocket, index: int) -> float:
    """Ideal (Tsiolkovsky) delta-v of one stage, carrying everything above it."""
    m0 = sum(s.dry_mass + s.prop_mass for s in rocket.stages[index:])
    m1 = m0 - rocket.stages[index].prop_mass
    return rocket.stages[index].isp * G0 * math.log(m0 / m1)


def total_delta_v(rocket: Rocket) -> float:
    return sum(stage_delta_v(rocket, i) for i in range(len(rocket.stages)))


def required_delta_v_estimate(planet: Planet) -> float:
    """Rough delta-v to reach a low circular orbit just above the atmosphere.

    Circular velocity plus a crude 15% + 700 m/s allowance for gravity,
    drag, and steering losses. Good to maybe +-15%; use it for sizing, not truth.
    """
    return 1.15 * planet.circular_velocity(planet.space_altitude) + 700.0


# --- Presets -----------------------------------------------------------------
#
# "easy" is a small Kerbin-scale planet: same surface gravity as Earth but a
# 600 km radius, so orbital velocity is ~2.3 km/s instead of 7.8 km/s and a
# full orbit takes ~31 minutes instead of ~90. Episodes stay short and the
# delta-v budget is forgiving. "earth" is the real thing, for hard mode.

PLANETS: dict[str, Planet] = {
    "easy": Planet(
        name="easy",
        radius=600_000.0,
        mu=3.5316e12,            # gives 9.81 m/s^2 surface gravity
        rho0=1.225,
        scale_height=5_600.0,
        space_altitude=70_000.0,
    ),
    "earth": Planet(
        name="earth",
        radius=6_371_000.0,
        mu=3.986004418e14,
        rho0=1.225,
        scale_height=8_500.0,
        space_altitude=100_000.0,
    ),
}

ROCKETS: dict[str, Rocket] = {
    # ~6.9 km/s ideal delta-v vs ~3.3 km/s required: roomy margin on purpose.
    "easy": Rocket(
        name="easy",
        stages=(
            Stage(dry_mass=3_000.0, prop_mass=15_000.0, thrust=350_000.0,
                  isp=280.0, cd_area=8.0),
            Stage(dry_mass=1_000.0, prop_mass=2_000.0, thrust=60_000.0,
                  isp=320.0, cd_area=3.0),
        ),
        max_turn_rate=math.radians(15.0),
    ),
    # ~13 km/s ideal delta-v vs ~9.7 km/s required.
    "earth": Rocket(
        name="earth",
        stages=(
            Stage(dry_mass=22_000.0, prop_mass=400_000.0, thrust=7_500_000.0,
                  isp=288.0, cd_area=25.0),
            Stage(dry_mass=5_500.0, prop_mass=55_000.0, thrust=900_000.0,
                  isp=345.0, cd_area=8.0),
        ),
        max_turn_rate=math.radians(15.0),
    ),
}


def describe(rocket: Rocket, planet: Planet) -> str:
    lines = [
        f"Planet '{planet.name}': R={planet.radius/1000:.0f} km, "
        f"g={planet.surface_gravity:.2f} m/s^2, "
        f"atmosphere top / reward cutoff at {planet.space_altitude/1000:.0f} km",
        f"  circular velocity just above atmosphere: "
        f"{planet.circular_velocity(planet.space_altitude):.0f} m/s",
        f"  estimated delta-v to orbit (incl. losses): "
        f"~{required_delta_v_estimate(planet):.0f} m/s",
        f"Rocket '{rocket.name}': launch mass {rocket.launch_mass/1000:.1f} t, "
        f"liftoff TWR {rocket.stages[0].thrust / (rocket.launch_mass * planet.surface_gravity):.2f}",
    ]
    for i in range(len(rocket.stages)):
        s = rocket.stages[i]
        lines.append(
            f"  stage {i + 1}: {s.prop_mass/1000:.1f} t propellant, "
            f"Isp {s.isp:.0f} s, ideal delta-v {stage_delta_v(rocket, i):.0f} m/s"
        )
    lines.append(f"  total ideal delta-v: {total_delta_v(rocket):.0f} m/s")
    return "\n".join(lines)
