"""Pygame renderer for TwoStageToOrbitEnv.

Design notes:
- The camera eases between two framings: a rocket-following view (rocket a
  little below center, zoom keyed to altitude) and a whole-planet "map view"
  once the apoapsis leaves the atmosphere. Zoom and blend are smoothed every
  frame so nothing ever jumps.
- The atmosphere is drawn as concentric shells fading from sky blue to space
  black, so ascent reads as climbing through layers, and the planet gets a
  blue halo in map view. Stars live behind everything.
- The rocket is a real two-stage silhouette that shrinks gently as the view
  widens and visibly loses its booster at staging.
- When the current trajectory is an ellipse, it is drawn KSP-style with Ap/Pe
  markers.
- rgb_array mode draws to a plain Surface and never opens a window, so it
  works headless (tests, video capture) without a display.
- Secondary bodies are the env's `bodies` list of `CelestialBody` objects
  (ToMoon's moon; ToMars's star and target). Inside a body's sphere of
  influence the map view re-centres on it; once the rocket leaves the
  planet's own sphere in a star system, the view becomes a system view
  centred on the star with the planets' orbit circles.
"""

from __future__ import annotations

import math
import random
from typing import ClassVar

import numpy as np

W, H = 960, 720
ZOOM_STEP = 1.25  # per key press, wheel notch or button click
ZOOM_MIN, ZOOM_MAX = 0.1, 50.0

SPACE = (5, 7, 16)
SKY = (96, 154, 209)
STAR = (120, 126, 148)
GROUND = (56, 80, 58)
GROUND_DARK = (44, 62, 46)
GROUND_EDGE = (116, 146, 116)
PAD = (150, 152, 160)
PAD_DARK = (96, 98, 106)
SPACE_LINE = (96, 118, 190)
TRAIL = (110, 205, 215)
ORBIT_PATH = (168, 156, 235)
BODY = (232, 234, 240)
BODY_SHADE = (176, 182, 196)
BOOSTER = (196, 200, 210)
MOON = (190, 196, 208)
MOON_SHADE = (142, 150, 166)
MOON_SITE = (255, 214, 110)
MARS = (206, 122, 86)
MARS_SHADE = (150, 84, 60)
SUN = (255, 214, 96)
SUN_GLOW = (120, 96, 40)
SYSTEM_ORBIT = (70, 80, 110)
INTERSTAGE = (60, 64, 78)
FLAME_OUT = (255, 140, 40)
FLAME_IN = (255, 224, 120)
TEXT = (214, 220, 235)
TEXT_DIM = (150, 158, 178)
BAR_BG = (40, 45, 60)
FUEL = (118, 200, 122)
THROTTLE = (255, 170, 70)
GOOD = (98, 200, 120)
BAD = (235, 100, 90)
ACCENT = (120, 170, 255)


def _lerp(a, b, t):
    return a + (b - a) * t


def _mix(c1, c2, t):
    return tuple(round(_lerp(c1[i], c2[i], t)) for i in range(3))


class Renderer:
    def __init__(self, env, mode: str):
        import pygame

        self.pg = pygame
        self.env = env
        self.mode = mode
        pygame.font.init()
        self.font_big = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 21)
        self.font_small = pygame.font.Font(None, 17)
        self.surf = pygame.Surface((W, H))
        self.window = None
        self.clock = None
        if mode == "human":
            pygame.init()
            pygame.display.set_caption("TwoStageToOrbit")
            self.window = pygame.display.set_mode((W, H))
            self.clock = pygame.time.Clock()

        rng = random.Random(7)
        self.stars = [
            (rng.uniform(0, W), rng.uniform(0, H), rng.choice((1, 1, 1, 2))) for _ in range(130)
        ]
        # surface features near the pad: (arc offset m, half-width m, height m)
        self.hills = [
            (o, rng.uniform(250.0, 900.0), rng.uniform(40.0, 150.0))
            for o in (rng.uniform(-30000.0, 30000.0) for _ in range(34))
        ]
        self._span = None  # smoothed view half-width, m
        self._blend = 0.0  # smoothed rocket-view -> map-view factor
        self._fblend = 0.0  # smoothed planet-focus -> moon-focus factor
        self.zoom = 1.0  # the user's multiplier on the automatic framing

    # --- camera ---------------------------------------------------------------

    def _bodies(self):
        """The env's secondary `CelestialBody` objects (none for the pure
        launch-planet missions)."""
        return list(getattr(self.env, "bodies", []))

    def _soi(self):
        """The body whose sphere of influence the rocket is in:
        (name, center_x, center_y, mu, body_radius, body). Secondary bodies
        win over the planet (smallest sphere first); a star counts only once
        the rocket has left the planet's own sphere; `body` is None for the
        launch planet."""
        env = self.env
        t = env._time
        bodies = self._bodies()
        for body in sorted((b for b in bodies if b.kind != "star"), key=lambda b: b.soi):
            if body.in_soi(env._rx, env._ry, t):
                bx, by = body.position(t)
                return body.name, bx, by, body.mu, body.radius, body
        planet_soi = getattr(env, "planet_soi", math.inf)
        if math.hypot(env._rx, env._ry) > planet_soi:
            for body in bodies:
                if body.kind == "star":
                    bx, by = body.position(t)
                    return body.name, bx, by, body.mu, body.radius, body
        return "planet", 0.0, 0.0, env.planet.mu, env.planet.radius, None

    def _system_span(self) -> float:
        """Half-width of the whole-system view: the widest orbit around the star."""
        radii = list(getattr(self.env, "system_orbit_radii", []))
        return 1.05 * max(radii) if radii else 0.0

    def _camera(self):
        env = self.env
        p = env.planet
        c = p.space_altitude
        alt = max(0.0, env._altitude())
        peri, apo, _energy, _ = env._orbit_elements()
        apo_eff = min(apo if math.isfinite(apo) else 3.0 * p.radius, 3.0 * p.radius)

        # Pull out to the whole-planet map view only once the trajectory is
        # (nearly) orbital -- periapsis above the ground -- or genuinely high.
        # Blending on apoapsis would yank the camera out mid-ascent.
        blend_t = max(
            0.0,
            min(1.0, (peri + 0.15 * c) / (0.6 * c)),
            min(1.0, (alt - 0.9 * c) / (1.8 * c)),
        )
        span_rocket = max(1500.0, 900.0 + 1.9 * alt)
        span_map = 1.15 * (p.radius + max(apo_eff, c))

        # KSP-style: inside the moon's SOI the map view re-centers on the
        # moon and the span shrinks to moon scale. The focus factor is
        # smoothed so crossing the boundary glides instead of jumping;
        # at 1 the focus tracks the (moving) moon exactly, lag-free.
        soi, sx, sy, _, soi_body_r, body = self._soi()
        mx, my = sx, sy
        if body is not None and body.kind != "star":
            blend_t = 1.0
            d = math.hypot(env._rx - mx, env._ry - my)
            span_map = 1.15 * (soi_body_r + max(d - soi_body_r, 0.35 * soi_body_r))
        elif body is not None:
            # system view: the star at the centre, wide enough for the
            # widest orbit and for wherever the rocket has got to
            blend_t = 1.0
            d = math.hypot(env._rx - mx, env._ry - my)
            span_map = 1.15 * max(d, self._system_span())
        want_f = 1.0 if body is not None else 0.0
        if self._span is None:
            self._fblend = want_f  # first frame ever: start converged
            self._focus_name = soi
        elif getattr(self, "_focus_name", soi) != soi and soi != "planet":
            self._fblend = 0.0  # new focus body: glide out from the planet view
            self._focus_name = soi
        else:
            self._fblend += (want_f - self._fblend) * 0.06
            if soi != "planet":
                self._focus_name = soi
        fx, fy = self._fblend * mx, self._fblend * my

        rn = math.hypot(env._rx, env._ry)
        d_focus = math.hypot(env._rx - fx, env._ry - fy)
        if self._span is None:
            self._span = _lerp(span_rocket, span_map, blend_t)
            self._blend = blend_t
        else:
            self._blend += (blend_t - self._blend) * 0.08
            span_t = _lerp(span_rocket, span_map, self._blend)
            # never let the blend geometry push the rocket off-screen
            span_t = max(span_t, 1.9 * self._blend * d_focus)
            self._span += (span_t - self._span) * 0.12

        m = self._blend
        ux, uy = env._rx / rn, env._ry / rn
        bias = 0.18 * self._span * (1.0 - m)
        cx = _lerp(env._rx + ux * bias, fx, m)
        cy = _lerp(env._ry + uy * bias, fy, m)
        return cx, cy, (W / 2) / self.view_span

    @property
    def view_span(self) -> float:
        """Half-width of the view (m): the smoothed automatic framing over the
        user's zoom, so the zoom persists across the rocket and map views."""
        return (self._span if self._span is not None else 1500.0) / self.zoom

    # --- zoom -------------------------------------------------------------------

    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, float(zoom)))

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.zoom * factor)

    def _zoom_buttons(self):
        """The on-screen buttons: (label, rect, factor); factor None resets."""
        pg = self.pg
        x, y, s = W - 44, H - 118, 30
        return [
            ("+", pg.Rect(x, y, s, s), ZOOM_STEP),
            ("-", pg.Rect(x, y + s + 6, s, s), 1.0 / ZOOM_STEP),
            ("1", pg.Rect(x, y + 2 * (s + 6), s, s), None),
        ]

    def handle_event(self, event) -> bool:
        """Zoom from a pygame event: + / = / keypad + zoom in, - / keypad -
        zoom out, 0 resets; the mouse wheel; a click on the on-screen
        buttons. Returns whether the event was consumed. In human mode the
        mouse events are read by `draw()` itself; scripts that drain the
        event queue (fly_manual) forward every event here."""
        pg = self.pg
        if event.type == pg.KEYDOWN:
            if event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.zoom_by(ZOOM_STEP)
            elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                self.zoom_by(1.0 / ZOOM_STEP)
            elif event.key in (pg.K_0, pg.K_KP0):
                self.set_zoom(1.0)
            else:
                return False
            return True
        if event.type == pg.MOUSEWHEEL:
            self.zoom_by(ZOOM_STEP if event.y > 0 else 1.0 / ZOOM_STEP)
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for _label, rect, factor in self._zoom_buttons():
                if rect.collidepoint(event.pos):
                    if factor is None:
                        self.set_zoom(1.0)
                    else:
                        self.zoom_by(factor)
                    return True
        return False

    def _draw_zoom_controls(self):
        pg = self.pg
        for label, rect, _factor in self._zoom_buttons():
            pg.draw.rect(self.surf, BAR_BG, rect, border_radius=5)
            pg.draw.rect(self.surf, TEXT_DIM, rect, width=1, border_radius=5)
            t = self.font_big.render(label, True, TEXT)
            self.surf.blit(
                t, (rect.centerx - t.get_width() / 2, rect.centery - t.get_height() / 2 + 1)
            )
        if abs(self.zoom - 1.0) > 1e-9:
            t = self.font_small.render(f"zoom x{self.zoom:.2g}", True, TEXT_DIM)
            self.surf.blit(t, (W - 14 - t.get_width(), H - 122 - t.get_height()))

    def _to_screen(self, x, y, cam):
        cx, cy, ppm = cam
        return ((x - cx) * ppm + W / 2, -(y - cy) * ppm + H / 2)

    def _on_screen(self, pt, margin=2000.0):
        return -margin <= pt[0] <= W + margin and -margin <= pt[1] <= H + margin

    def _draw_runs(self, pts, color, cam):
        """Polyline of world points, split into on-screen runs (no clamping)."""
        run = []
        for x, y in pts:
            s = self._to_screen(x, y, cam)
            if self._on_screen(s):
                run.append(s)
            else:
                if len(run) > 1:
                    self.pg.draw.aalines(self.surf, color, False, run)
                run = []
        if len(run) > 1:
            self.pg.draw.aalines(self.surf, color, False, run)

    # --- world layers -----------------------------------------------------------

    def _surface_angles(self, cam, radius, extra=0.06):
        cx, cy, ppm = cam
        if math.hypot(cx, cy) < 1.0:
            ang0 = math.atan2(self.env._ry, self.env._rx)
        else:
            ang0 = math.atan2(cy, cx)
        half = 1.5 * (W / ppm) / radius + extra
        return ang0, min(math.pi, half)

    def _fill_below(self, radius, color, cam, edge=None):
        """Fill everything at or below `radius` from the planet center."""
        pg = self.pg
        ppm = cam[2]
        r_px = radius * ppm
        if r_px < 1.3 * max(W, H):
            center = self._to_screen(0.0, 0.0, cam)
            pg.draw.circle(self.surf, color, center, r_px)
            if edge:
                pg.draw.circle(self.surf, edge, center, r_px, width=2)
            return
        ang0, half = self._surface_angles(cam, radius)
        n = 72
        pts = [
            self._to_screen(radius * math.cos(a), radius * math.sin(a), cam)
            for a in (ang0 - half + 2 * half * i / n for i in range(n + 1))
        ]
        pts = [(max(-30000.0, min(30000.0, x)), max(-30000.0, min(30000.0, y))) for x, y in pts]
        inner = max(0.05 * radius, radius - 8.0 * (W / ppm))
        lo = [
            self._to_screen(inner * math.cos(a), inner * math.sin(a), cam)
            for a in (ang0 + half, ang0 - half)
        ]
        lo = [(max(-30000.0, min(30000.0, x)), max(-30000.0, min(30000.0, y))) for x, y in lo]
        pg.draw.polygon(self.surf, color, pts + lo)
        if edge:
            pg.draw.aalines(self.surf, edge, False, pts)

    def _draw_atmosphere(self, cam):
        p = self.env.planet
        top = 1.25 * p.space_altitude
        n = 20
        for i in reversed(range(n)):
            alt = top * ((i + 1) / n) ** 1.35
            color = _mix(SKY, SPACE, (alt / top) ** 0.65)
            self._fill_below(p.radius + alt, color, cam)

    def _draw_space_line(self, cam):
        p = self.env.planet
        radius = p.radius + p.space_altitude
        ang0, half = self._surface_angles(cam, radius)
        n = max(48, int(half * 200))
        pts = [
            self._to_screen(radius * math.cos(a), radius * math.sin(a), cam)
            for a in (ang0 - half + 2 * half * i / n for i in range(n + 1))
        ]
        for i in range(0, n, 2):  # dashed
            a, b = pts[i], pts[i + 1]
            if self._on_screen(a, margin=100) and self._on_screen(b, margin=100):
                self.pg.draw.aaline(self.surf, SPACE_LINE, a, b)
        # label above the rocket's longitude
        env = self.env
        ang = math.atan2(env._ry, env._rx)
        lp = self._to_screen(radius * math.cos(ang), radius * math.sin(ang), cam)
        if self._on_screen(lp, margin=-20):
            label = self.font_small.render(
                f"space  {p.space_altitude / 1000:.0f} km", True, SPACE_LINE
            )
            self.surf.blit(label, (lp[0] + 8, lp[1] - 18))

    def _draw_surface_features(self, cam):
        pg = self.pg
        p = self.env.planet
        ppm = cam[2]
        R = p.radius
        if ppm < 0.012:  # features would be sub-pixel noise; mark the pad instead
            s = self._to_screen(0.0, R, cam)
            if self._on_screen(s, margin=0):
                pg.draw.circle(self.surf, PAD, s, 3)
                self.surf.blit(self.font_small.render("pad", True, TEXT_DIM), (s[0] + 6, s[1] - 6))
            return

        def at(arc_m, up_m):
            a = math.pi / 2 + arc_m / R  # pad is at angle pi/2
            r = R + up_m
            return self._to_screen(r * math.cos(a), r * math.sin(a), cam)

        for off, hw, hh in self.hills:
            tri = [at(off - hw, 0.0), at(off, hh), at(off + hw, 0.0)]
            if any(self._on_screen(t, margin=200) for t in tri):
                pg.draw.polygon(self.surf, GROUND_DARK, tri)
        # launch pad: a slab the rocket stands on, plus a small tower
        deck = [at(-70, 4), at(70, 4), at(70, -24), at(-70, -24)]
        if any(self._on_screen(t, margin=200) for t in deck):
            pg.draw.polygon(self.surf, PAD, deck)
            pg.draw.polygon(self.surf, PAD_DARK, [at(-58, 64), at(-46, 64), at(-46, 4), at(-58, 4)])

    # --- trajectory -------------------------------------------------------------

    def _orbit_path(self):
        """Trajectory conic in the frame of the SOI body, KSP-style: near
        the moon it is moon-relative and drawn around the moon. Bound,
        reasonably-round orbits draw as the full ellipse; hyperbolic and
        near-parabolic paths draw as the arc ahead of the rocket -- an
        arriving lander is hyperbolic relative to the moon right up to the
        capture burn, and it should still see where it is headed."""
        env = self.env
        _soi_name, bx, by, mu, body_r, body = self._soi()
        rx, ry = env._rx - bx, env._ry - by
        if body is not None and body.kind != "star":
            mvx, mvy = body.velocity(env._time)
            vx, vy = env._vx - mvx, env._vy - mvy
            min_apo = 0.05 * body_r  # any bound orbit about the body is interesting
            r_lim = body.soi
        elif body is not None:
            svx, svy = body.velocity(env._time)
            vx, vy = env._vx - svx, env._vy - svy
            min_apo = 0.05 * body_r
            r_lim = 3.0 * self._system_span()
        else:
            vx, vy = env._vx, env._vy
            min_apo = 0.3 * env.planet.space_altitude
            r_lim = 3.2 * env.planet.radius
        r = math.hypot(rx, ry)
        v2 = vx * vx + vy * vy
        energy = 0.5 * v2 - mu / r
        rv = rx * vx + ry * vy
        ex = ((v2 - mu / r) * rx - rv * vx) / mu
        ey = ((v2 - mu / r) * ry - rv * vy) / mu
        e = math.hypot(ex, ey)
        w = math.atan2(ey, ex)
        h = rx * vy - ry * vx
        p_semi = h * h / mu
        peri_alt = p_semi / (1.0 + e) - body_r if e > 1e-9 else r - body_r
        pe = (bx + (peri_alt + body_r) * math.cos(w), by + (peri_alt + body_r) * math.sin(w))

        if energy < 0.0 and e <= 0.985:
            a = -mu / (2.0 * energy)
            if a * (1.0 + e) - body_r < min_apo:
                return None  # too small to be interesting yet
            pts = []
            for k in range(97):
                nu = 2.0 * math.pi * k / 96
                rr = a * (1.0 - e * e) / (1.0 + e * math.cos(nu))
                pts.append((bx + rr * math.cos(w + nu), by + rr * math.sin(w + nu)))
            return pts, pe, pts[48], peri_alt, a * (1.0 + e) - body_r

        # open conic: sample true anomaly from the rocket forward until the
        # path leaves the SOI (or the map). The world angle is w + sgn*nu,
        # sgn from the angular momentum -- these missions fly westward.
        if e < 1e-9:
            return None
        sgn = 1.0 if h >= 0.0 else -1.0
        nu0 = math.acos(max(-1.0, min(1.0, (p_semi / r - 1.0) / e)))
        if rv < 0.0:
            nu0 = -nu0
        nu_lim = math.acos(max(-1.0, min(1.0, (p_semi / r_lim - 1.0) / e)))
        if e > 1.0:
            nu_lim = min(nu_lim, math.acos(-1.0 / e) - 0.02)
        if nu_lim <= nu0:
            return None
        pts = []
        for k in range(129):
            nu = nu0 + (nu_lim - nu0) * k / 128
            rr = p_semi / (1.0 + e * math.cos(nu))
            if rr < 0.0 or rr > 1.05 * r_lim:
                break
            pts.append((bx + rr * math.cos(w + sgn * nu), by + rr * math.sin(w + sgn * nu)))
            if rr < body_r:
                break  # ends at the surface: no ghost path out the far side
        if len(pts) < 2:
            return None
        return pts, pe, (0.0, 0.0), peri_alt, -1.0

    def _draw_orbit(self, cam):
        path = self._orbit_path()
        if path is None:
            return
        pts, pe, ap, peri_alt, apo_alt = path
        self._draw_runs(pts, ORBIT_PATH, cam)
        for pos, alt, name in ((ap, apo_alt, "Ap"), (pe, peri_alt, "Pe")):
            if alt <= 0.0:
                continue
            s = self._to_screen(*pos, cam)
            if self._on_screen(s, margin=-10):
                self.pg.draw.circle(self.surf, ORBIT_PATH, s, 4)
                lbl = self.font_small.render(f"{name} {alt / 1000:.0f}k", True, ORBIT_PATH)
                self.surf.blit(lbl, (s[0] + 6, s[1] - 16))

    def _draw_trail(self, cam):
        trail = self.env._trail
        pts = trail[:: max(1, len(trail) // 1200)] + trail[-1:]
        n = len(pts)
        if n < 2:
            return
        prev = None
        for i, (x, y) in enumerate(pts):
            s = self._to_screen(x, y, cam)
            ok = self._on_screen(s)
            if prev is not None and ok:
                color = _mix((70, 100, 128), TRAIL, 0.3 + 0.7 * (i / n))
                self.pg.draw.aaline(self.surf, color, prev, s)
            prev = s if ok else None

    BODY_COLORS: ClassVar[dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
        "moon": (MOON, MOON_SHADE),
        "planet": (MARS, MARS_SHADE),
    }

    def _draw_star(self, body, cam):
        """A star, and the orbit circles of the system around it (only when
        the view is wide enough to show them)."""
        env = self.env
        center = self._to_screen(*body.position(env._time), cam)
        ppm = cam[2]
        for radius in getattr(env, "system_orbit_radii", []):
            r_px = radius * ppm
            if 20 < r_px < 20000 and self._on_screen(center, margin=r_px + 50):
                self.pg.draw.circle(self.surf, SYSTEM_ORBIT, center, r_px, width=1)
        if self._on_screen(center, margin=400):
            r_px = max(4.0, body.radius * ppm)
            self.pg.draw.circle(self.surf, SUN_GLOW, center, r_px * 1.8)
            self.pg.draw.circle(self.surf, SUN, center, r_px)

    def _draw_secondary_bodies(self, cam):
        """Every `CelestialBody` the env has: stars with their system orbits,
        moons and planets with their sphere of influence and landing site."""
        env = self.env
        t = env._time
        site_angle = getattr(env, "_site_angle", None)
        for body in self._bodies():
            if body.kind == "star":
                self._draw_star(body, cam)
                continue
            fill, shade = self.BODY_COLORS.get(body.kind, (MOON, MOON_SHADE))
            center = self._to_screen(*body.position(t), cam)
            if not self._on_screen(center, margin=200):
                continue
            r_px = max(3.0, body.radius * cam[2])
            self.pg.draw.circle(self.surf, fill, center, r_px)
            self.pg.draw.circle(self.surf, shade, center, r_px, width=2)
            soi_px = body.soi * cam[2]
            if 1.4 * r_px < soi_px < 2500:
                self.pg.draw.circle(self.surf, SPACE_LINE, center, soi_px, width=1)
            if site_angle is None:
                continue
            site = self._to_screen(*body.site_world(t, site_angle), cam)
            if self._on_screen(site, margin=100):
                self.pg.draw.circle(self.surf, MOON_SITE, site, 4)
                self.surf.blit(
                    self.font_small.render("site", True, MOON_SITE), (site[0] + 6, site[1] - 6)
                )

    # --- rocket -----------------------------------------------------------------

    def _frame_fn(self, world_pos, theta, size, cam):
        """Maps rocket-frame coords (+x = nose, units of `size`) to screen."""
        pos = self._to_screen(*world_pos, cam)
        c, s = math.cos(theta), math.sin(theta)

        def at(fx, fy):
            fx, fy = fx * size, fy * size
            return (pos[0] + fx * c - fy * s, pos[1] - (fx * s + fy * c))

        return at

    def _draw_booster(self, at):
        """Stage-1 booster with fins and interstage; base (engine) at x=0."""
        pg = self.pg
        pg.draw.polygon(
            self.surf, BOOSTER, [at(0.0, 0.21), at(1.10, 0.21), at(1.10, -0.21), at(0.0, -0.21)]
        )
        pg.draw.polygon(self.surf, BOOSTER, [at(0.0, 0.21), at(0.45, 0.21), at(-0.06, 0.52)])
        pg.draw.polygon(self.surf, BOOSTER, [at(0.0, -0.21), at(0.45, -0.21), at(-0.06, -0.52)])
        pg.draw.polygon(
            self.surf,
            INTERSTAGE,
            [at(1.10, 0.19), at(1.24, 0.19), at(1.24, -0.19), at(1.10, -0.19)],
        )

    def _draw_upper(self, at, x0):
        """Upper stage body + nose cone starting at rocket-frame x0."""
        pg = self.pg
        pg.draw.polygon(
            self.surf,
            BODY,
            [at(x0, 0.18), at(x0 + 1.03, 0.18), at(x0 + 1.03, -0.18), at(x0, -0.18)],
        )
        pg.draw.polygon(
            self.surf,
            BODY_SHADE,
            [at(x0, -0.09), at(x0 + 1.03, -0.09), at(x0 + 1.03, -0.18), at(x0, -0.18)],
        )
        pg.draw.polygon(
            self.surf, BODY, [at(x0 + 1.02, 0.185), at(x0 + 1.47, 0.0), at(x0 + 1.02, -0.185)]
        )

    def _draw_rocket(self, cam):
        pg = self.pg
        env = self.env
        size = max(9.0, min(24.0, 24.0 * (1500.0 / self.view_span) ** 0.22))

        d = env._debris
        if d is not None:
            dpos = self._to_screen(d["rx"], d["ry"], cam)
            if self._on_screen(dpos, margin=100):
                self._draw_booster(self._frame_fn((d["rx"], d["ry"]), d["theta"], size * 0.95, cam))

        at = self._frame_fn((env._rx, env._ry), env._theta, size, cam)
        if env._stage == 0:
            self._draw_booster(at)
            self._draw_upper(at, 1.24)
        else:
            self._draw_upper(at, 0.0)
        if self._blend > 0.5:  # ship marker: easy to spot at planet scale
            pg.draw.circle(self.surf, ACCENT, self._to_screen(env._rx, env._ry, cam), 10, width=1)

        throttle = getattr(env, "_throttle_actual", 0.0)
        if throttle > 0.0 and env._terminal is None:
            flick = 1.0 + 0.22 * (np.random.random() - 0.5)
            L = (0.6 + 1.9 * throttle) * flick
            pg.draw.polygon(self.surf, FLAME_OUT, [at(0.02, 0.15), at(0.02, -0.15), at(-L, 0.0)])
            pg.draw.polygon(
                self.surf, FLAME_IN, [at(0.02, 0.09), at(0.02, -0.09), at(-0.55 * L, 0.0)]
            )

    # --- HUD --------------------------------------------------------------------

    def _panel(self, w, h, alpha=150):
        p = self.pg.Surface((w, h), self.pg.SRCALPHA)
        p.fill((10, 13, 24, alpha))
        return p

    def _headline(self, body) -> dict[str, float]:
        """The HUD's altitude, speed, vr and vt. Inside a secondary body's
        sphere of influence they are body-relative, matching the frame the
        info elements switched to; between bodies (ToMars) heliocentric,
        with the distance to the star in the altitude slot; otherwise the
        env's own planet-frame numbers."""
        env = self.env
        if body is None:
            info = env._info()
            return {
                "alt_km": info["altitude"] / 1000,
                "speed": info["speed"],
                "vr": info["vr"],
                "vt": info["vt"],
            }
        args = (env._rx, env._ry, env._vx, env._vy, env._time)
        vr, vt = body.radial_tangential(*args)
        if body.kind == "star":
            alt = body.distance(env._rx, env._ry, env._time)
        else:
            alt = body.altitude(env._rx, env._ry, env._time)
        return {"alt_km": alt / 1000, "speed": body.relative_speed(*args), "vr": vr, "vt": vt}

    def _draw_hud(self):
        pg = self.pg
        env = self.env
        info = env._info()

        panel = self._panel(258, 168)
        self.surf.blit(panel, (10, 10))
        soi_name, _bx, _by, _mu, _r, body = self._soi()
        in_moon = body is not None and body.kind != "star"
        in_sun = body is not None and body.kind == "star"
        head = self._headline(body)
        self.surf.blit(self.font_big.render(f"{head['alt_km']:8.1f} km", True, TEXT), (24, 20))
        self.surf.blit(self.font_big.render(f"{head['speed']:6.0f} m/s", True, TEXT), (24, 52))
        apo = info["apoapsis"]
        apo_s = "--" if not math.isfinite(apo) else f"{apo / 1000:.1f}"
        peri_s = "--" if info["periapsis"] < 0 else f"{info['periapsis'] / 1000:.1f}"
        rows = [
            (f"vr {head['vr']:+7.0f}", f"vt {head['vt']:+7.0f}"),
            (f"Ap {apo_s:>8}", f"Pe {peri_s:>8}"),
            (f"dv {info['dv_remaining']:6.0f}", f"t  {info['time']:6.0f} s"),
            (
                f"stage  {info['stage'] + 1}/{len(env.rocket.stages)}",
                f"R  {info['episode_return']:7.1f}",
            ),
        ]
        y = 88
        for left, right in rows:
            self.surf.blit(self.font.render(left, True, TEXT_DIM), (24, y))
            self.surf.blit(self.font.render(right, True, TEXT_DIM), (134, y))
            y += 21

        # right side: fuel + throttle bars
        panel = self._panel(160, 24 + 22 * (len(env.rocket.stages) + 1))
        self.surf.blit(panel, (W - 170, 10))
        y = 22
        for i, st in enumerate(env.rocket.stages):
            frac = env._fuel / st.prop_mass if i == env._stage else (0.0 if i < env._stage else 1.0)
            self.surf.blit(self.font_small.render(f"S{i + 1}", True, TEXT_DIM), (W - 162, y + 1))
            pg.draw.rect(self.surf, BAR_BG, (W - 138, y, 118, 12), border_radius=3)
            if frac > 0:
                pg.draw.rect(self.surf, FUEL, (W - 138, y, 118 * frac, 12), border_radius=3)
            y += 22
        self.surf.blit(self.font_small.render("TH", True, TEXT_DIM), (W - 162, y + 1))
        pg.draw.rect(self.surf, BAR_BG, (W - 138, y, 118, 12), border_radius=3)
        thr = getattr(env, "_throttle_actual", 0.0)
        if thr > 0:
            pg.draw.rect(self.surf, THROTTLE, (W - 138, y, 118 * thr, 12), border_radius=3)

        # status badge
        badge_text = None
        if env._terminal is None and in_moon:
            badge_text = f"{soi_name.upper()} SOI", ACCENT
        elif env._terminal is None and in_sun:
            badge_text = "INTERPLANETARY", ACCENT
        elif env._terminal is None and info.get("returning"):
            badge_text = f"RETURN   {info['pad_downrange'] / 1000:.0f} km to pad", ACCENT
        elif env._terminal is None and info["in_space"]:
            if info["orbit"]:
                text = "ORBIT"
                if "orbit_progress" in info:
                    text = f"ORBIT   hold {info['orbit_progress']:.0%}"
                badge_text = text, GOOD
            else:
                badge_text = "IN SPACE", ACCENT
        if badge_text is not None:
            text, color = badge_text
            badge = self.font.render(text, True, color)
            bw = badge.get_width() + 20
            panel = self._panel(bw, 26, alpha=170)
            self.surf.blit(panel, (W / 2 - bw / 2, 12))
            self.surf.blit(badge, (W / 2 - badge.get_width() / 2, 17))

        if env._terminal is not None:
            msgs = {
                "crashed": ("CRASHED", BAD),
                "landed": ("LANDED", TEXT),
                "landed_on_pad": ("LANDED ON THE PAD", GOOD),
                "landed_moon": ("LANDED ON THE MOON", GOOD),
                "landed_moon_site": ("LANDED ON THE SITE", GOOD),
                "landed_mars": ("LANDED ON MARS", GOOD),
                "landed_mars_site": ("LANDED ON THE SITE", GOOD),
                "escaped": ("ESCAPED THE PLANET", ORBIT_PATH),
                "orbit": ("STABLE ORBIT", GOOD),
                "time_limit": ("TIME'S UP", TEXT),
            }
            text, color = msgs.get(env._terminal, (env._terminal.upper(), TEXT))
            banner = self.font_big.render(text, True, color)
            sub_text = f"return {info['episode_return']:.1f}"
            if info.get("returning") and env._terminal in ("landed", "landed_on_pad", "crashed"):
                sub_text += f"   {info['pad_downrange'] / 1000:.1f} km from pad"
            sub = self.font.render(sub_text, True, TEXT_DIM)
            bw = max(banner.get_width(), sub.get_width()) + 48
            panel = self._panel(bw, 74, alpha=190)
            self.surf.blit(panel, (W / 2 - bw / 2, H / 2 - 60))
            self.surf.blit(banner, (W / 2 - banner.get_width() / 2, H / 2 - 48))
            self.surf.blit(sub, (W / 2 - sub.get_width() / 2, H / 2 - 14))

    # --- frame ------------------------------------------------------------------

    def draw(self):
        pg = self.pg
        cam = self._camera()

        self.surf.fill(SPACE)
        for x, y, r in self.stars:
            pg.draw.circle(self.surf, STAR, (x, y), r)
        self._draw_atmosphere(cam)
        self._fill_below(self.env.planet.radius, GROUND, cam, edge=GROUND_EDGE)
        self._draw_surface_features(cam)
        self._draw_space_line(cam)
        self._draw_orbit(cam)
        self._draw_secondary_bodies(cam)
        self._draw_trail(cam)
        self._draw_rocket(cam)
        self._draw_hud()

        if self.mode == "human":
            for event in pg.event.get(eventtype=[pg.MOUSEWHEEL, pg.MOUSEBUTTONDOWN]):
                self.handle_event(event)
            self._draw_zoom_controls()
            self.window.blit(self.surf, (0, 0))
            pg.event.pump()
            pg.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])
            return None
        frame = pg.surfarray.array3d(self.surf)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        if self.window is not None:
            self.pg.display.quit()
            self.pg.quit()
