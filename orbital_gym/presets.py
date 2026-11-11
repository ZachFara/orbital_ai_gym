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
class Moon:
    name: str
    radius: float          # m
    mu: float              # m^3/s^2
    orbit_radius: float    # m, circular orbit around the planet center
    orbit_period: float    # s
    site_angle: float = 0.0  # rad, site angle around moon body in moon frame


@dataclass(frozen=True)
class Star:
    name: str
    radius: float   # m
    mu: float       # m^3/s^2


@dataclass(frozen=True)
class OrbitingBody:
    """A planet on circular rails around a star: a landing target with its own
    (possibly thin) atmosphere. Used for the Mars-like target of ToMars."""
    name: str
    radius: float          # m
    mu: float              # m^3/s^2
    orbit_radius: float    # m, around the star
    orbit_period: float    # s (Keplerian for the star's mu)
    site_angle: float = 0.0  # rad, landing site in the body frame
    rho0: float = 0.0        # kg/m^3 sea-level density (0 = airless)
    scale_height: float = 1.0
    space_altitude: float = 0.0  # m, top of the atmosphere

    def density(self, altitude: float) -> float:
        if self.rho0 <= 0.0 or altitude >= self.space_altitude:
            return 0.0
        return self.rho0 * math.exp(-max(0.0, altitude) / self.scale_height)


@dataclass(frozen=True)
class StarSystem:
    """The launch planet on circular rails around a star, plus a target body.

    home_phase0 / target_phase0 are the bodies' angles around the star at
    t = 0. The default target phase is chosen so that a Hohmann departure
    window opens shortly after a prompt ascent-and-hold (see ToMarsEnv).
    """
    name: str
    star: Star
    home_orbit_radius: float
    home_orbit_period: float
    target: OrbitingBody
    home_phase0: float = 0.0
    target_phase0: float = 0.0


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

MOONS: dict[str, Moon] = {
    # Gameplay moons (short-period transfers suitable for RL episodes).
    # Periods are Keplerian for the orbit radius (T = 2*pi*sqrt(r^3/mu)).
    # Rails at any other speed make the moon-relative frame non-inertial:
    # a lander fights a phantom v^2/r - mu_p/r^2 acceleration (the old
    # arbitrary 7200 s period put 1.2 m/s^2 of it -- 14% of moon surface
    # gravity, and most of local gravity up at braking altitude -- under
    # every landing, and made anything co-moving with the moon
    # planet-hyperbolic).
    "easy": Moon(
        name="easy_moon",
        radius=160_000.0,
        mu=2.4e11,
        orbit_radius=2_400_000.0,
        orbit_period=2.0 * math.pi * (2_400_000.0**3 / PLANETS["easy"].mu) ** 0.5,
        site_angle=0.0,
    ),
    "earth": Moon(
        name="earth_moon_arcade",
        radius=700_000.0,
        mu=8.0e12,
        orbit_radius=18_000_000.0,
        orbit_period=2.0 * math.pi * (18_000_000.0**3 / PLANETS["earth"].mu) ** 0.5,
        site_angle=0.0,
    ),
}

# --- a compact star system for the Mars mission -------------------------------
#
# Scaled so that the whole trip fits an episode of ~12,000 one-second steps:
# the "easy" planet circles a small, heavy star at 17,100 km (year 3.9 h),
# the Mars-like target at 26,000 km (the real Earth/Mars ratio of 1.52).
# A Hohmann transfer takes ~2.8 h; departure needs ~1.0 km/s from the
# 670 km parking orbit, arrival ~1.7 km/s at the target's surface (its own
# escape velocity dominates). The star's pull at the planet is 3.4 m/s^2, so
# its tidal field is a real perturbation on a long parking-orbit hold: the
# planet's Hill radius is only ~3 planet radii. The planet is 0.35% of the
# star's mass (3.5 Jupiters): heavy enough to bend heliocentric conics a
# little, which is why the pilot trims its cruise.
_SUN_MU = 1.0e15
_HOME_R = 1.71e7
_MARS_R = 2.6e7

STARS: dict[str, Star] = {
    "easy": Star(name="easy_sun", radius=1.2e6, mu=_SUN_MU),
}

SYSTEMS: dict[str, StarSystem] = {
    "easy": StarSystem(
        name="easy_system",
        star=STARS["easy"],
        home_orbit_radius=_HOME_R,
        home_orbit_period=2.0 * math.pi * (_HOME_R**3 / _SUN_MU) ** 0.5,
        target=OrbitingBody(
            name="easy_mars",
            radius=340_000.0,
            mu=4.28e11,              # 3.7 m/s^2 surface gravity
            orbit_radius=_MARS_R,
            orbit_period=2.0 * math.pi * (_MARS_R**3 / _SUN_MU) ** 0.5,
            site_angle=0.0,
            rho0=0.015,              # thin: aerobraking helps, barely
            scale_height=9_000.0,
            space_altitude=50_000.0,
        ),
        home_phase0=0.0,
        # Hohmann lead angle at departure is ~44 deg and shrinks 0.012 deg/s;
        # 72 deg at t=0 puts the window ~40 min in: after the ascent and a
        # quarter-revolution hold, with most of one parking orbit to spare.
        target_phase0=math.radians(72.0),
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
    # ~10 km/s ideal delta-v: the moon-mission rocket (ToMoon default on
    # "easy"). Arrival braking costs ~1.8 km/s (the moon's own escape
    # velocity dominates), and on the 6.9 km/s "easy" rocket the ascent,
    # injection, arrival, and descent losses close only on a perfect
    # flight, if at all. Liftoff TWR and stage-1 burn time match "easy" so
    # the same ascent profile flies; the vacuum stage is the tug that pays
    # for injection, capture, and landing.
    "moon": Rocket(
        name="moon",
        stages=(
            Stage(dry_mass=6_200.0, prop_mass=31_000.0, thrust=740_000.0,
                  isp=280.0, cd_area=13.0),
            Stage(dry_mass=1_000.0, prop_mass=6_000.0, thrust=137_000.0,
                  isp=350.0, cd_area=5.0),
        ),
        max_turn_rate=math.radians(15.0),
    ),
    # Three stages, ~11.7 km/s ideal: the Mars-mission rocket (ToMars default).
    # Ascent ~4.3, departure ~1.0, arrival + landing ~2.0 plus losses leaves
    # a few km/s for midcourse corrections and a sloppy descent. Liftoff TWR
    # and stage-1 burn time match "easy" so the same gravity turn flies.
    "mars": Rocket(
        name="mars",
        stages=(
            Stage(dry_mass=7_000.0, prop_mass=36_000.0, thrust=950_000.0,
                  isp=285.0, cd_area=14.0),
            Stage(dry_mass=1_500.0, prop_mass=9_000.0, thrust=160_000.0,
                  isp=340.0, cd_area=5.0),
            Stage(dry_mass=700.0, prop_mass=2_600.0, thrust=40_000.0,
                  isp=350.0, cd_area=3.0),
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
