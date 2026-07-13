"""Pygame renderer for TwoStageToOrbitEnv.

Camera follows the rocket near the ground and pulls back to a whole-planet
view as it climbs, so both the pad wobble and the final orbit are visible.
rgb_array mode draws to a plain Surface and never opens a window, so it works
headless (tests, video capture) without a display.
"""

from __future__ import annotations

import math

import numpy as np

W, H = 960, 720

SPACE = (6, 8, 18)
SKY = (110, 160, 210)
GROUND = (52, 74, 54)
GROUND_EDGE = (110, 140, 110)
SPACE_LINE = (70, 90, 150)
TRAIL = (90, 190, 200)
ROCKET = (235, 235, 235)
FLAME = (255, 150, 40)
TEXT = (210, 215, 230)
BAR_BG = (50, 55, 70)
BAR_FUEL = (120, 200, 120)


def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, env, mode: str):
        import pygame

        self.pg = pygame
        self.env = env
        self.mode = mode
        pygame.font.init()
        self.font = pygame.font.Font(None, 22)
        self.surf = pygame.Surface((W, H))
        self.window = None
        self.clock = None
        if mode == "human":
            pygame.init()
            pygame.display.set_caption("TwoStageToOrbit")
            self.window = pygame.display.set_mode((W, H))
            self.clock = pygame.time.Clock()

    # --- camera ---------------------------------------------------------------

    def _camera(self):
        env = self.env
        p = env.planet
        alt = env._altitude()
        peri, apo, energy, _ = env._orbit_elements()
        span = max(2500.0, 3.0 * alt)  # view half-width, m (rocket view)
        blend = min(1.0, max(0.0, (alt - 0.20 * p.radius) / (0.30 * p.radius)))
        if blend > 0.0:
            reach = max(alt, 0.0 if not math.isfinite(apo) else apo)
            reach = min(reach, 3.0 * p.radius)
            span_planet = 1.45 * (p.radius + max(reach, p.space_altitude))
            span = _lerp(span, span_planet, blend)
        cx = _lerp(env._rx, 0.0, blend)
        cy = _lerp(env._ry, 0.0, blend)
        ppm = (W / 2) / span
        return cx, cy, ppm

    def _to_screen(self, x, y, cam, clamp=True):
        cx, cy, ppm = cam
        sx = (x - cx) * ppm + W / 2
        sy = -(y - cy) * ppm + H / 2
        if clamp:
            return (max(-30000.0, min(30000.0, sx)), max(-30000.0, min(30000.0, sy)))
        return (sx, sy)

    # --- drawing ---------------------------------------------------------------

    def draw(self):
        pg = self.pg
        env = self.env
        p = env.planet
        cam = self._camera()
        alt = env._altitude()

        t = min(1.0, max(0.0, alt / p.space_altitude))
        bg = tuple(round(_lerp(SKY[i], SPACE[i], t)) for i in range(3))
        self.surf.fill(bg)

        self._draw_ring(p.radius, GROUND, cam, fill=True)
        self._draw_ring(p.radius + p.space_altitude, SPACE_LINE, cam, fill=False)

        # draw only on-screen runs; clamping far-away points would paint
        # false straight lines across the view
        run = []
        for x, y in env._trail[::4] + env._trail[-1:]:
            sx, sy = self._to_screen(x, y, cam, clamp=False)
            if -2000.0 <= sx <= W + 2000.0 and -2000.0 <= sy <= H + 2000.0:
                run.append((sx, sy))
            else:
                if len(run) > 1:
                    pg.draw.aalines(self.surf, TRAIL, False, run)
                run = []
        if len(run) > 1:
            pg.draw.aalines(self.surf, TRAIL, False, run)

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

    def _draw_ring(self, radius, color, cam, fill):
        """Planet surface or atmosphere line, drawn as circle or local arc."""
        pg = self.pg
        env = self.env
        cx, cy, ppm = cam
        r_px = radius * ppm
        if r_px < 1.2 * max(W, H):
            center = self._to_screen(0.0, 0.0, cam)
            if fill:
                pg.draw.circle(self.surf, color, center, r_px)
                pg.draw.circle(self.surf, GROUND_EDGE, center, r_px, width=2)
            else:
                pg.draw.circle(self.surf, color, center, r_px, width=1)
            return
        # zoomed in: draw the nearby arc as a polyline / filled band
        ang0 = math.atan2(env._ry, env._rx)
        half = (W / ppm) / radius + 0.05
        n = 64
        arc = [
            (radius * math.cos(a), radius * math.sin(a))
            for a in (ang0 - half + 2 * half * i / n for i in range(n + 1))
        ]
        pts = [self._to_screen(x, y, cam) for x, y in arc]
        if fill:
            inner = max(0.05 * radius, radius - 6 * (W / ppm))
            lo = [
                self._to_screen(inner * math.cos(a), inner * math.sin(a), cam)
                for a in (ang0 + half, ang0 - half)
            ]
            pg.draw.polygon(self.surf, color, pts + lo)
            pg.draw.aalines(self.surf, GROUND_EDGE, False, pts)
        else:
            pg.draw.aalines(self.surf, color, False, pts)

    def _draw_rocket(self, cam):
        pg = self.pg
        env = self.env
        pos = self._to_screen(env._rx, env._ry, cam)
        th = env._theta
        c, s = math.cos(th), math.sin(th)

        def body(fx, fy):  # rocket frame: +x = nose
            return (pos[0] + (fx * c - fy * s), pos[1] - (fx * s + fy * c))

        size = 14
        nose = body(size, 0)
        left = body(-0.6 * size, 0.5 * size)
        right = body(-0.6 * size, -0.5 * size)
        pg.draw.polygon(self.surf, ROCKET, [nose, left, right])
        throttle = getattr(env, "_throttle_actual", 0.0)
        if throttle > 0.0:
            flick = 1.0 + 0.25 * (np.random.random() - 0.5)
            L = size * (0.8 + 1.6 * throttle) * flick
            tip = body(-0.6 * size - L, 0)
            fl = body(-0.6 * size, 0.28 * size)
            fr = body(-0.6 * size, -0.28 * size)
            pg.draw.polygon(self.surf, FLAME, [tip, fl, fr])

    def _draw_hud(self):
        env = self.env
        info = env._info()
        apo = info["apoapsis"]
        apo_s = "inf" if not math.isfinite(apo) else f"{apo / 1000:7.1f} km"
        lines = [
            f"t {info['time']:7.0f} s   stage {info['stage'] + 1}/{len(env.rocket.stages)}",
            f"alt {info['altitude'] / 1000:8.1f} km   vel {info['speed']:6.0f} m/s",
            f"vr  {info['vr']:7.0f} m/s   vt {info['vt']:6.0f} m/s",
            f"apo {apo_s}   peri {info['periapsis'] / 1000:7.1f} km",
            f"dv left {info['dv_remaining']:5.0f} m/s",
            "IN SPACE" + ("  *ORBIT*" if info["orbit"] else "") if info["in_space"] else "",
        ]
        y = 8
        for line in lines:
            if line:
                self.surf.blit(self.font.render(line, True, TEXT), (10, y))
            y += 20
        for i, s in enumerate(env.rocket.stages):
            frac = env._fuel / s.prop_mass if i == env._stage else (0.0 if i < env._stage else 1.0)
            self.pg.draw.rect(self.surf, BAR_BG, (W - 130, 10 + 18 * i, 120, 12))
            self.pg.draw.rect(self.surf, BAR_FUEL, (W - 130, 10 + 18 * i, 120 * frac, 12))

    def close(self):
        if self.window is not None:
            self.pg.display.quit()
            self.pg.quit()
