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
"""

from __future__ import annotations

import math
import random

import numpy as np

W, H = 960, 720

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
            (rng.uniform(0, W), rng.uniform(0, H), rng.choice((1, 1, 1, 2)))
            for _ in range(130)
        ]
        # surface features near the pad: (arc offset m, half-width m, height m)
        self.hills = [
            (o, rng.uniform(250.0, 900.0), rng.uniform(40.0, 150.0))
            for o in (rng.uniform(-30000.0, 30000.0) for _ in range(34))
        ]
        self._span = None   # smoothed view half-width, m
        self._blend = 0.0   # smoothed rocket-view -> map-view factor

    # --- camera ---------------------------------------------------------------

    def _camera(self):
        env = self.env
        p = env.planet
        c = p.space_altitude
        alt = max(0.0, env._altitude())
        peri, apo, energy, _ = env._orbit_elements()
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

        rn = math.hypot(env._rx, env._ry)
        if self._span is None:
            self._span, self._blend = span_rocket, blend_t
        else:
            self._blend += (blend_t - self._blend) * 0.08
            span_t = _lerp(span_rocket, span_map, self._blend)
            # never let the blend geometry push the rocket off-screen
            span_t = max(span_t, 1.9 * self._blend * rn)
            self._span += (span_t - self._span) * 0.12

        m = self._blend
        ux, uy = env._rx / rn, env._ry / rn
        bias = 0.18 * self._span * (1.0 - m)
        cx = _lerp(env._rx + ux * bias, 0.0, m)
        cy = _lerp(env._ry + uy * bias, 0.0, m)
        return cx, cy, (W / 2) / self._span

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
        if ppm < 0.012:  # features would be sub-pixel noise
            return
        R = p.radius

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
            pg.draw.polygon(self.surf, PAD_DARK,
                            [at(-58, 64), at(-46, 64), at(-46, 4), at(-58, 4)])

    # --- trajectory -------------------------------------------------------------

    def _orbit_path(self):
        env = self.env
        p = env.planet
        rx, ry, vx, vy = env._rx, env._ry, env._vx, env._vy
        r = math.hypot(rx, ry)
        v2 = vx * vx + vy * vy
        energy = 0.5 * v2 - p.mu / r
        if energy >= 0.0:
            return None
        rv = rx * vx + ry * vy
        ex = ((v2 - p.mu / r) * rx - rv * vx) / p.mu
        ey = ((v2 - p.mu / r) * ry - rv * vy) / p.mu
        e = math.hypot(ex, ey)
        if e > 0.985:
            return None
        a = -p.mu / (2.0 * energy)
        if a * (1.0 + e) - p.radius < 0.3 * p.space_altitude:
            return None  # too small to be interesting yet
        w = math.atan2(ey, ex)
        pts = []
        for k in range(97):
            nu = 2.0 * math.pi * k / 96
            rr = a * (1.0 - e * e) / (1.0 + e * math.cos(nu))
            pts.append((rr * math.cos(w + nu), rr * math.sin(w + nu)))
        pe = pts[0]
        ap = pts[48]
        return pts, pe, ap, a * (1.0 - e) - p.radius, a * (1.0 + e) - p.radius

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
        pg.draw.polygon(self.surf, BOOSTER,
                        [at(0.0, 0.21), at(1.10, 0.21), at(1.10, -0.21), at(0.0, -0.21)])
        pg.draw.polygon(self.surf, BOOSTER,
                        [at(0.0, 0.21), at(0.45, 0.21), at(-0.06, 0.52)])
        pg.draw.polygon(self.surf, BOOSTER,
                        [at(0.0, -0.21), at(0.45, -0.21), at(-0.06, -0.52)])
        pg.draw.polygon(self.surf, INTERSTAGE,
                        [at(1.10, 0.19), at(1.24, 0.19), at(1.24, -0.19), at(1.10, -0.19)])

    def _draw_upper(self, at, x0):
        """Upper stage body + nose cone starting at rocket-frame x0."""
        pg = self.pg
        pg.draw.polygon(self.surf, BODY,
                        [at(x0, 0.18), at(x0 + 1.03, 0.18), at(x0 + 1.03, -0.18), at(x0, -0.18)])
        pg.draw.polygon(self.surf, BODY_SHADE,
                        [at(x0, -0.09), at(x0 + 1.03, -0.09), at(x0 + 1.03, -0.18), at(x0, -0.18)])
        pg.draw.polygon(self.surf, BODY,
                        [at(x0 + 1.02, 0.185), at(x0 + 1.47, 0.0), at(x0 + 1.02, -0.185)])

    def _draw_rocket(self, cam):
        pg = self.pg
        env = self.env
        size = max(9.0, min(24.0, 24.0 * (1500.0 / self._span) ** 0.22))

        d = env._debris
        if d is not None:
            dpos = self._to_screen(d["rx"], d["ry"], cam)
            if self._on_screen(dpos, margin=100):
                self._draw_booster(
                    self._frame_fn((d["rx"], d["ry"]), d["theta"], size * 0.95, cam))

        at = self._frame_fn((env._rx, env._ry), env._theta, size, cam)
        if env._stage == 0:
            self._draw_booster(at)
            self._draw_upper(at, 1.24)
        else:
            self._draw_upper(at, 0.0)
        if self._blend > 0.5:  # ship marker: easy to spot at planet scale
            pg.draw.circle(self.surf, ACCENT,
                           self._to_screen(env._rx, env._ry, cam), 10, width=1)

        throttle = getattr(env, "_throttle_actual", 0.0)
        if throttle > 0.0 and env._terminal is None:
            flick = 1.0 + 0.22 * (np.random.random() - 0.5)
            L = (0.6 + 1.9 * throttle) * flick
            pg.draw.polygon(self.surf, FLAME_OUT,
                            [at(0.02, 0.15), at(0.02, -0.15), at(-L, 0.0)])
            pg.draw.polygon(self.surf, FLAME_IN,
                            [at(0.02, 0.09), at(0.02, -0.09), at(-0.55 * L, 0.0)])

    # --- HUD --------------------------------------------------------------------

    def _panel(self, w, h, alpha=150):
        p = self.pg.Surface((w, h), self.pg.SRCALPHA)
        p.fill((10, 13, 24, alpha))
        return p

    def _draw_hud(self):
        pg = self.pg
        env = self.env
        info = env._info()

        panel = self._panel(258, 168)
        self.surf.blit(panel, (10, 10))
        alt_km = info["altitude"] / 1000
        self.surf.blit(self.font_big.render(f"{alt_km:8.1f} km", True, TEXT), (24, 20))
        self.surf.blit(self.font_big.render(f"{info['speed']:6.0f} m/s", True, TEXT), (24, 52))
        apo = info["apoapsis"]
        apo_s = "--" if not math.isfinite(apo) else f"{apo / 1000:.1f}"
        peri_s = "--" if info["periapsis"] < 0 else f"{info['periapsis'] / 1000:.1f}"
        rows = [
            (f"vr {info['vr']:+7.0f}", f"vt {info['vt']:+7.0f}"),
            (f"Ap {apo_s:>8}", f"Pe {peri_s:>8}"),
            (f"dv {info['dv_remaining']:6.0f}", f"t  {info['time']:6.0f} s"),
            (f"stage  {info['stage'] + 1}/{len(env.rocket.stages)}",
             f"R  {info['episode_return']:7.1f}"),
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
            frac = (env._fuel / st.prop_mass if i == env._stage
                    else (0.0 if i < env._stage else 1.0))
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
        if env._terminal is None and info["in_space"]:
            text, color = ("ORBIT", GOOD) if info["orbit"] else ("IN SPACE", ACCENT)
            badge = self.font.render(text, True, color)
            bw = badge.get_width() + 20
            panel = self._panel(bw, 26, alpha=170)
            self.surf.blit(panel, (W / 2 - bw / 2, 12))
            self.surf.blit(badge, (W / 2 - badge.get_width() / 2, 17))

        if env._terminal is not None:
            msgs = {
                "crashed": ("CRASHED", BAD),
                "landed": ("LANDED", TEXT),
                "escaped": ("ESCAPED THE PLANET", ORBIT_PATH),
                "orbit": ("STABLE ORBIT", GOOD),
                "time_limit": ("TIME'S UP", TEXT),
            }
            text, color = msgs.get(env._terminal, (env._terminal.upper(), TEXT))
            banner = self.font_big.render(text, True, color)
            sub = self.font.render(
                f"return {info['episode_return']:.1f}", True, TEXT_DIM)
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
        self._draw_trail(cam)
        self._draw_rocket(cam)
        self._draw_hud()

        if self.mode == "human":
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
