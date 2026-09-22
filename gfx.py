"""Neon look for a native 1920x1080 canvas: palette, fonts, camera, additive glow sprites, world
rendering (grid, walls, pickups, projectiles, agents, trails), particles and UI drawing helpers.

UI layouts are written in a 960x540 design space and scaled with z()/R()/P(), so text is always
rendered at native resolution (nothing is upscaled)."""
import math
import random

import numpy as np
import pygame as pg

import config as C

SW, SH = 1920, 1080          # canvas (fullscreen SCALED -> native on 1080p screens)
U = SW / 960                 # UI scale factor for the 960x540 design grid


def z(v):
    return int(round(v * U))


def R(x, y, w, h):
    return pg.Rect(z(x), z(y), z(w), z(h))


def P(x, y):
    return (z(x), z(y))


HUD_H = z(46)

# ------------------------------------------------------------------ palette
BG0, BG1 = (6, 7, 15), (14, 15, 32)
GRID, GRID_HI = (22, 25, 52), (38, 42, 88)
TEXT, DIM, FAINT = (236, 240, 255), (138, 148, 186), (70, 78, 112)
CYAN, PINK, GOLD, RED = (0, 229, 255), (255, 46, 136), (255, 204, 77), (255, 70, 85)
PLAYER_C = CYAN
ENEMY_C = {"beginner": (70, 250, 160), "easy": (255, 222, 80), "medium": (255, 140, 50),
           "hard": (255, 46, 120), "impossible": (170, 90, 255)}
WALL_EDGE = (120, 110, 255)
ITEM_C = [(255, 150, 60), (255, 225, 80), (190, 110, 255), (80, 200, 255), (90, 255, 140)]
ITEM_LABEL = ["SÖRÉTES", "GÉPFEGYVER", "RAILGUN", "PAJZS", "ÉLET"]
WEAPON_LABEL = ["ALAP", "SÖRÉTES", "GÉPFEGYVER", "RAILGUN"]
WEAPON_C = [None, ITEM_C[0], ITEM_C[1], ITEM_C[2]]
SHIELD_C = ITEM_C[3]


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def lighten(c, k):
    return mix(c, (255, 255, 255), k)


def darken(c, k):
    return mix(c, (0, 0, 0), k)


# ------------------------------------------------------------------ fonts & text
_fonts = {}


def font(size, bold=True):
    """size is in design units (scaled to the canvas)."""
    key = (size, bold)
    if key not in _fonts:
        f = None
        for name in ("bahnschrift", "segoeui", "arial"):
            if pg.font.match_font(name):
                f = pg.font.SysFont(name, z(size), bold=bold)
                break
        _fonts[key] = f or pg.font.Font(None, z(size))
    return _fonts[key]


def mono(size):
    return pg.font.SysFont("consolas", z(size))


def render_spaced(s, f, color, spacing=0):
    spacing = z(spacing)
    if not spacing:
        return f.render(s, True, color)
    # SDL_ttf shifts a lone accented capital down so its accent fits; rendering every glyph next
    # to the same tall reference character keeps them all on one baseline.
    ref = "Ő"
    rw = f.size(ref)[0]
    glyphs = []
    for ch in s:
        full = f.render(ref + ch, True, color)
        cw = max(1, full.get_width() - rw)
        glyphs.append(full.subsurface((rw, 0, cw, full.get_height())))
    w = sum(g.get_width() for g in glyphs) + spacing * (len(s) - 1)
    h = max(g.get_height() for g in glyphs)
    out = pg.Surface((max(w, 1), h), pg.SRCALPHA)
    x = 0
    for g in glyphs:
        out.blit(g, (x, 0))
        x += g.get_width() + spacing
    return out


def place(img, center=None, topleft=None, topright=None, midleft=None, midright=None, midtop=None):
    r = img.get_rect()
    if center is not None:
        r.center = center
    elif topright is not None:
        r.topright = topright
    elif midleft is not None:
        r.midleft = midleft
    elif midright is not None:
        r.midright = midright
    elif midtop is not None:
        r.midtop = midtop
    else:
        r.topleft = topleft
    return r


def text(surf, s, f, color, spacing=0, shadow=True, alpha=255, **pos):
    img = render_spaced(s, f, color, spacing)
    r = place(img, **pos)
    if shadow:
        sh = render_spaced(s, f, (0, 0, 0), spacing)
        sh.set_alpha(int(alpha * 0.6))
        surf.blit(sh, r.move(0, z(1.5)))
    if alpha < 255:
        img.set_alpha(alpha)
    surf.blit(img, r)
    return r


def glow_source(img):
    s = pg.Surface(img.get_size())
    s.fill((0, 0, 0))
    s.blit(img, (0, 0))
    return s


def soft_blur(img, k=4):
    """Gaussian blur with padding so the glow isn't clipped."""
    w, h = img.get_size()
    pad = pg.Surface((w + 8 * k, h + 8 * k), pg.SRCALPHA if img.get_flags() & pg.SRCALPHA else 0)
    pad.fill((0, 0, 0, 0))
    pad.blit(img, (4 * k, 4 * k))
    if hasattr(pg.transform, "gaussian_blur"):
        return pg.transform.gaussian_blur(pad, 2 * k)
    small = pg.transform.smoothscale(pad, (max(1, pad.get_width() // k), max(1, pad.get_height() // k)))
    return pg.transform.smoothscale(small, pad.get_size())


def fast_glow(src, k):
    """Soft bloom of an opaque RGB image: blur a 1/4-size copy and scale it back up. Visually the
    same as a full-size gaussian blur for a glow, but ~15x cheaper (big titles no longer stall a frame)."""
    s = 4
    w, h = src.get_size()
    pad = pg.Surface((w + 8 * k, h + 8 * k))
    pad.fill((0, 0, 0))
    pad.blit(src, (4 * k, 4 * k))
    small = pg.transform.smoothscale(pad, (max(1, pad.get_width() // s), max(1, pad.get_height() // s)))
    if hasattr(pg.transform, "gaussian_blur"):
        small = pg.transform.gaussian_blur(small, max(1, 2 * k // s))
    return pg.transform.smoothscale(small, pad.get_size())


_glow_text_cache = {}


def glow_text(surf, s, f, color, spacing=0, strength=1.0, alpha=255, glow=None, **pos):
    """Crisp text over a soft neon bloom (cached)."""
    gc = glow or color
    key = (s, id(f), color, gc, spacing)
    if key not in _glow_text_cache:
        if len(_glow_text_cache) > 400:
            _glow_text_cache.clear()
        img = render_spaced(s, f, color, spacing)
        _glow_text_cache[key] = (img, fast_glow(glow_source(render_spaced(s, f, gc, spacing)), max(3, f.get_height() // 10)))
    img, blur = _glow_text_cache[key]
    r = place(img, **pos)
    br = blur.get_rect(center=r.center)
    alpha = max(0, min(255, int(alpha)))
    k = alpha / 255 * strength
    if k < 0.999:
        blur = blur.copy()
        blur.fill((int(255 * min(k, 1)),) * 3, special_flags=pg.BLEND_RGB_MULT)
    if alpha < 255:
        img = img.copy()
        img.set_alpha(alpha)
    surf.blit(blur, br, special_flags=pg.BLEND_RGB_ADD)
    sh = render_spaced(s, f, (0, 0, 0), spacing)
    sh.set_alpha(int(alpha * 0.5))
    surf.blit(sh, r.move(0, z(1.5)))
    surf.blit(img, r)
    return r


def wrap(s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.size(t)[0] <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ glow sprites
_glow_cache = {}


def glow(color, radius, power=2.2, intensity=1.0):
    """Radial falloff sprite for BLEND_RGB_ADD blits."""
    radius = max(2, int(radius))
    key = (color, radius, power, round(intensity, 2))
    if key not in _glow_cache:
        if len(_glow_cache) > 3000:
            _glow_cache.clear()
        n = 2 * radius
        y, x = np.mgrid[0:n, 0:n] + 0.5
        d = np.sqrt((x - radius) ** 2 + (y - radius) ** 2) / radius
        a = np.clip(1 - d, 0, 1) ** power * intensity
        arr = (np.array(color)[None, None, :] * a[..., None]).clip(0, 255).astype(np.uint8)
        _glow_cache[key] = pg.surfarray.make_surface(arr.transpose(1, 0, 2))
    return _glow_cache[key]


def add_glow(surf, color, x, y, radius, intensity=1.0, power=2.2):
    # quantise size/brightness so particles reuse a small set of cached sprites
    radius = max(2, int(radius) // 2 * 2)
    g = glow(color, radius, power, round(min(intensity, 1.5) * 20) / 20)
    surf.blit(g, (x - radius, y - radius), special_flags=pg.BLEND_RGB_ADD)


# ------------------------------------------------------------------ camera
class Camera:
    """World -> screen transform with smooth follow, zoom, recoil kick and shake."""

    def __init__(self, zoom=1.0):
        self.cx, self.cy = C.W / 2, C.H / 2
        self.zoom = self.target_zoom = zoom
        self.kx = self.ky = 0.0      # recoil kick (screen px)
        self.shake = 0.0
        self.shake_on = True
        self.sx = self.sy = 0.0

    def view_size(self):
        return SW / self.zoom, SH / self.zoom

    def clamp(self):
        vw, vh = self.view_size()
        self.cx = C.W / 2 if vw >= C.W else min(max(self.cx, vw / 2), C.W - vw / 2)
        top = HUD_H / self.zoom          # the top of the map may scroll under the HUD bar
        self.cy = C.H / 2 if vh >= C.H + top else min(max(self.cy, vh / 2 - top), C.H - vh / 2)

    def snap(self, x, y):
        self.cx, self.cy = x, y
        self.clamp()

    def follow(self, x, y, speed=0.12):
        self.zoom += (self.target_zoom - self.zoom) * 0.08
        self.cx += (x - self.cx) * speed
        self.cy += (y - self.cy) * speed
        self.clamp()

    def kick(self, angle, amount):
        self.kx -= math.cos(angle) * amount
        self.ky -= math.sin(angle) * amount

    def add_shake(self, a):
        if self.shake_on:
            self.shake = min(self.shake + a, 22)

    def update(self):
        self.kx *= 0.78
        self.ky *= 0.78
        self.shake *= 0.86
        if self.shake > 0.4:
            self.sx, self.sy = random.uniform(-1, 1) * self.shake, random.uniform(-1, 1) * self.shake
        else:
            self.sx = self.sy = 0.0

    def w2s(self, x, y):
        return ((x - self.cx) * self.zoom + SW / 2 + self.kx + self.sx,
                (y - self.cy) * self.zoom + SH / 2 + self.ky + self.sy)

    def s2w(self, x, y):
        return ((x - SW / 2 - self.kx - self.sx) / self.zoom + self.cx,
                (y - SH / 2 - self.ky - self.sy) / self.zoom + self.cy)

    def visible(self, x, y, margin=60):
        sx, sy = self.w2s(x, y)
        return -margin < sx < SW + margin and -margin < sy < SH + margin


class ScreenCam:
    """Identity camera for effects drawn in screen space (menus, results)."""
    zoom = 1.0

    def w2s(self, x, y):
        return x, y

    def visible(self, x, y, margin=60):
        return True


# ------------------------------------------------------------------ backgrounds
_bg_cache = {}


def backdrop():
    if "backdrop" not in _bg_cache:
        w, h = SW // 4, SH // 4
        y, x = np.mgrid[0:h, 0:w]
        d = np.sqrt(((x - w * 0.5) / w) ** 2 + ((y - h * 0.45) / h) ** 2)
        t = np.clip(1 - d * 1.6, 0, 1) ** 1.5
        arr = np.stack([lerp(BG0[i], BG1[i] + 10, t) for i in range(3)], -1).astype(np.uint8)
        small = pg.surfarray.make_surface(arr.transpose(1, 0, 2))
        _bg_cache["backdrop"] = pg.transform.smoothscale(small, (SW, SH))
    return _bg_cache["backdrop"]


def vignette(color=(255, 30, 60)):
    key = ("vig", color)
    if key not in _bg_cache:
        w, h = SW // 4, SH // 4
        y, x = np.mgrid[0:h, 0:w]
        d = np.maximum(np.abs(x - w / 2) / (w / 2), np.abs(y - h / 2) / (h / 2))
        a = (np.clip((d - 0.55) / 0.45, 0, 1) ** 2 * 200).astype(np.uint8)
        s = pg.Surface((w, h), pg.SRCALPHA)
        s.fill((*color, 0))
        pg.surfarray.pixels_alpha(s)[:] = a.T
        _bg_cache[key] = pg.transform.smoothscale(s, (SW, SH))
    return _bg_cache[key]


def draw_floor(surf, cam, t=0):
    """Void outside the arena, floor, world-space grid and the neon arena border."""
    surf.blit(backdrop(), (0, 0))
    x0, y0 = cam.w2s(0, 0)
    x1, y1 = cam.w2s(C.W, C.H)
    floor = pg.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    pg.draw.rect(surf, (10, 11, 24), floor)
    step = 60
    start = max(0, int((-x0) / cam.zoom // step) * step)
    for gx in range(start, C.W + 1, step):
        sx = x0 + gx * cam.zoom
        if sx > SW:
            break
        pg.draw.line(surf, GRID_HI if gx % (step * 4) == 0 else GRID, (sx, max(y0, 0)), (sx, min(y1, SH)))
    start = max(0, int((-y0) / cam.zoom // step) * step)
    for gy in range(start, C.H + 1, step):
        sy = y0 + gy * cam.zoom
        if sy > SH:
            break
        pg.draw.line(surf, GRID_HI if gy % (step * 4) == 0 else GRID, (max(x0, 0), sy), (min(x1, SW), sy))
    edge = mix(CYAN, PINK, 0.5 + 0.5 * math.sin(t * 0.02))
    for w_, c in ((z(5), darken(edge, 0.8)), (z(3), darken(edge, 0.55)), (z(1), lighten(edge, 0.2))):
        pg.draw.rect(surf, c, floor.inflate(w_, w_), w_, border_radius=z(4))


_wall_cache = {}


def draw_walls(surf, rects, cam):
    zq = round(cam.zoom * 20) / 20
    for x0, y0, x1, y1 in rects:
        sx, sy = cam.w2s(x0, y0)
        w, h = (x1 - x0) * cam.zoom, (y1 - y0) * cam.zoom
        if sx > SW or sy > SH or sx + w < 0 or sy + h < 0:
            continue
        key = (int((x1 - x0) * zq), int((y1 - y0) * zq))
        if key not in _wall_cache:
            if len(_wall_cache) > 300:
                _wall_cache.clear()
            g = pg.Surface((key[0] + 60, key[1] + 60))
            g.fill((0, 0, 0))
            pg.draw.rect(g, WALL_EDGE, (30, 30, key[0], key[1]), 4, border_radius=6)
            _wall_cache[key] = soft_blur(g, 5)
        gs = _wall_cache[key]
        surf.blit(gs, gs.get_rect(center=(sx + w / 2, sy + h / 2)), special_flags=pg.BLEND_RGB_ADD)
        r = pg.Rect(int(sx), int(sy), int(w), int(h))
        pg.draw.rect(surf, (18, 18, 42), r, border_radius=6)
        inner = r.inflate(-z(5), -z(5))
        if inner.w > 6 and inner.h > 6:
            pg.draw.rect(surf, (28, 27, 62), inner, border_radius=4)
        pg.draw.rect(surf, lighten(WALL_EDGE, 0.35), r, max(2, int(2 * cam.zoom)), border_radius=6)


def draw_item_icon(surf, kind, x, y, r=11, t=0.0, glow_on=True):
    c = ITEM_C[kind]
    if glow_on:
        add_glow(surf, c, x, y, r * 3.2, intensity=0.55 + 0.2 * math.sin(t * 0.1))
    pg.draw.circle(surf, darken(c, 0.7), (int(x), int(y)), int(r))
    pg.draw.circle(surf, lighten(c, 0.25), (int(x), int(y)), int(r), max(2, int(r / 6)))
    w = lighten(c, 0.6)
    lw = max(2, int(r / 5))
    if kind == 0:      # shotgun: fan of three
        for a in (-0.45, 0, 0.45):
            sx, sy = x - r * 0.45, y + r * 0.1
            pg.draw.line(surf, w, (sx, sy), (sx + math.cos(a) * r * 0.95, sy + math.sin(a) * r * 0.95), lw)
    elif kind == 1:    # rapid: three bullets
        for i in (-1, 0, 1):
            pg.draw.circle(surf, w, (int(x + i * r * 0.42), int(y)), max(2, int(r / 5)))
    elif kind == 2:    # rail: long beam
        pg.draw.line(surf, w, (x - r * 0.6, y + r * 0.35), (x + r * 0.6, y - r * 0.35), lw + 1)
    elif kind == 3:    # shield: hexagon
        pts = [(x + math.cos(a) * r * 0.55, y + math.sin(a) * r * 0.55)
               for a in np.arange(6) * math.pi / 3 + math.pi / 6]
        pg.draw.polygon(surf, w, pts, lw)
    else:              # heal: plus
        pg.draw.line(surf, w, (x - r * 0.5, y), (x + r * 0.5, y), lw + 1)
        pg.draw.line(surf, w, (x, y - r * 0.5), (x, y + r * 0.5), lw + 1)


def draw_world(surf, snap, colors, cam, flash=None, hp_pips=False, t=0, trails=None):
    """Everything in world space, through `cam`. trails: per-agent list of recent positions."""
    zm = cam.zoom
    draw_floor(surf, cam, t)
    if len(snap.obstacles):
        draw_walls(surf, snap.obstacles, cam)
    if getattr(snap, "iact", None) is not None:
        for j in np.flatnonzero(snap.iact):
            x, y = snap.ipos[j]
            if cam.visible(x, y):
                sx, sy = cam.w2s(x, y + 2.5 * math.sin(t * 0.08 + j * 2))
                draw_item_icon(surf, int(snap.itype[j]), sx, sy, C.ITEM_R * zm, t)
    if trails is not None:
        for k, tr in enumerate(trails):
            if k >= len(snap.alive) or not snap.alive[k] or len(tr) < 2:
                continue
            c = colors[k]
            for i, (x, y) in enumerate(list(tr)[:-1]):
                a = (i + 1) / len(tr)
                sx, sy = cam.w2s(x, y)
                add_glow(surf, c, sx, sy, C.AGENT_R * zm * (0.6 + 0.6 * a), intensity=0.14 * a)
    pdmg = getattr(snap, "pdmg", None)
    for k in range(len(snap.pos)):
        c = colors[k]
        for j in np.flatnonzero(snap.pact[k]):
            wx, wy = snap.ppos[k, j]
            if not cam.visible(wx, wy):
                continue
            px, py = cam.w2s(wx, wy)
            v = snap.pvel[k, j] * zm
            big = pdmg is not None and pdmg[k, j] > 1
            add_glow(surf, c, px, py, (24 if big else 15) * zm, intensity=1.0 if big else 0.9)
            pg.draw.line(surf, darken(c, 0.35), (px - v[0] * 3.2, py - v[1] * 3.2), (px, py),
                         max(2, int((6 if big else 4) * zm)))
            pg.draw.line(surf, lighten(c, 0.5), (px - v[0] * 1.4, py - v[1] * 1.4), (px, py),
                         max(2, int((3 if big else 2) * zm)))
            pg.draw.circle(surf, (255, 255, 255), (int(px), int(py)), max(2, int((4 if big else 3) * zm)))
    weapon = getattr(snap, "weapon", None)
    shield = getattr(snap, "shield", None)
    for k in range(len(snap.pos)):
        if not snap.alive[k]:
            continue
        c = colors[k]
        hot = flash is not None and flash[k] > 0
        x, y = cam.w2s(*snap.pos[k])
        r = C.AGENT_R * zm
        pulse = 0.75 + 0.25 * math.sin(t * 0.12 + k)
        add_glow(surf, c, x, y, 46 * zm, intensity=(1.0 if hot else 0.55 * pulse))
        if shield is not None and shield[k] > 0:
            add_glow(surf, SHIELD_C, x, y, r + 22 * zm, intensity=0.35 + 0.12 * shield[k])
            pg.draw.circle(surf, lighten(SHIELD_C, 0.3), (int(x), int(y)), int(r + 8 * zm), max(1, int((1 + shield[k]) * zm)))
        body = lighten(c, 0.8) if hot else darken(c, 0.55)
        pg.draw.circle(surf, body, (int(x), int(y)), int(r))
        pg.draw.circle(surf, lighten(c, 0.25), (int(x), int(y)), int(r), max(2, int(3 * zm)))
        pg.draw.circle(surf, lighten(c, 0.7), (int(x), int(y)), max(2, int(4 * zm)))
        a = snap.aim[k]
        ca, sa = math.cos(a), math.sin(a)
        wk = int(weapon[k]) if weapon is not None else 0
        wc = WEAPON_C[wk] if wk > 0 else None
        tip_len = r + (18 if wc else 13) * zm
        tip = (x + ca * tip_len, y + sa * tip_len)
        b = r + 5 * zm
        b1 = (x + ca * b - sa * 6 * zm, y + sa * b + ca * 6 * zm)
        b2 = (x + ca * b + sa * 6 * zm, y + sa * b - ca * 6 * zm)
        pg.draw.polygon(surf, lighten(wc or c, 0.35), (tip, b1, b2))
        frac = 1 - snap.cd[k] / max(C.W_COOLDOWN[wk], 1)
        if frac < 1:
            rr = r + 7 * zm
            pg.draw.arc(surf, darken(c, 0.2), (x - rr, y - rr, 2 * rr, 2 * rr),
                        math.pi / 2, math.pi / 2 + math.tau * max(frac, 0), max(2, int(2 * zm)))
        if hp_pips and k != 0:
            n = int(snap.max_hp[k])
            w = 7 * zm
            x0 = x - (n * (w + 3 * zm) - 3 * zm) / 2
            for i in range(n):
                col = lighten(c, 0.2) if i < snap.hp[k] else (40, 44, 70)
                pg.draw.rect(surf, col, (x0 + i * (w + 3 * zm), y - r - 16 * zm, w, 4 * zm), border_radius=2)


def draw_minimap(surf, snap, colors, cam, rect):
    """Top-down overview with the camera's view rectangle."""
    panel(surf, rect, border=FAINT, alpha=170, radius=z(6))
    kx, ky = rect.w / C.W, rect.h / C.H
    for x0, y0, x1, y1 in snap.obstacles:
        pg.draw.rect(surf, darken(WALL_EDGE, 0.2), (rect.x + x0 * kx, rect.y + y0 * ky,
                                                    max(2, (x1 - x0) * kx), max(2, (y1 - y0) * ky)))
    if getattr(snap, "iact", None) is not None:
        for j in np.flatnonzero(snap.iact):
            x, y = snap.ipos[j]
            pg.draw.circle(surf, ITEM_C[int(snap.itype[j])], (int(rect.x + x * kx), int(rect.y + y * ky)), z(2))
    for k in range(len(snap.pos)):
        if snap.alive[k]:
            x, y = snap.pos[k]
            pg.draw.circle(surf, colors[k], (int(rect.x + x * kx), int(rect.y + y * ky)), z(3.5 if k == 0 else 3))
    vw, vh = cam.view_size()
    v = pg.Rect(rect.x + (cam.cx - vw / 2) * kx, rect.y + (cam.cy - vh / 2) * ky, vw * kx, vh * ky).clip(rect)
    pg.draw.rect(surf, (200, 210, 255), v, 1)


def offscreen_arrows(surf, snap, colors, cam, player=0):
    """Edge arrows pointing at living enemies outside the view."""
    cx, cy = SW / 2, (SH + HUD_H) / 2
    m = z(26)
    for k in range(len(snap.pos)):
        if k == player or not snap.alive[k]:
            continue
        x, y = cam.w2s(*snap.pos[k])
        if m <= x <= SW - m and HUD_H + m <= y <= SH - m:
            continue
        a = math.atan2(y - cy, x - cx)
        ca, sa = math.cos(a), math.sin(a)
        tx = abs(((SW - m if ca > 0 else m) - cx) / (ca if abs(ca) > 1e-6 else 1e-6))
        ty = abs(((SH - m if sa > 0 else HUD_H + m) - cy) / (sa if abs(sa) > 1e-6 else 1e-6))
        tt = min(tx, ty)
        ex, ey = cx + ca * tt, cy + sa * tt
        s = z(12)
        pts = [(ex + ca * s, ey + sa * s), (ex + math.cos(a + 2.5) * s, ey + math.sin(a + 2.5) * s),
               (ex + math.cos(a - 2.5) * s, ey + math.sin(a - 2.5) * s)]
        add_glow(surf, colors[k], ex, ey, z(22), intensity=0.6)
        pg.draw.polygon(surf, lighten(colors[k], 0.3), pts)


def map_preview(rects, w, h, color=WALL_EDGE):
    s = pg.Surface((w, h), pg.SRCALPHA)
    pg.draw.rect(s, (12, 13, 28, 255), s.get_rect(), border_radius=z(6))
    for gx in range(0, w, max(6, w // 12)):
        pg.draw.line(s, (24, 26, 52), (gx, 0), (gx, h))
    kx, ky = w / C.W, h / C.H
    for x0, y0, x1, y1 in rects:
        pg.draw.rect(s, lighten(color, 0.2), (x0 * kx, y0 * ky, max(2, (x1 - x0) * kx), max(2, (y1 - y0) * ky)),
                     border_radius=2)
    pg.draw.rect(s, (60, 60, 110, 255), s.get_rect(), 1, border_radius=z(6))
    return s


# ------------------------------------------------------------------ effects (world or screen space)
class Particles:
    """Additive glowing particles; 'spark' particles are drawn as streaks. Coordinates live in the
    space of the camera passed to draw()."""

    def __init__(self):
        self.p = []  # [x, y, vx, vy, life, max_life, color, size, spark]

    def burst(self, x, y, color, n=18, speed=4.0, life=30, size=3, spread=math.tau, angle=0.0, spark=False):
        for _ in range(n):
            a = angle + random.uniform(-spread / 2, spread / 2)
            s = speed * random.uniform(0.3, 1.0)
            lf = life * random.uniform(0.5, 1.0)
            self.p.append([x, y, math.cos(a) * s, math.sin(a) * s, lf, lf, color, size, spark])

    def update(self):
        for q in self.p:
            q[0] += q[2]
            q[1] += q[3]
            q[2] *= 0.91
            q[3] *= 0.91
            q[4] -= 1
        self.p = [q for q in self.p if q[4] > 0]

    def draw(self, surf, cam):
        zm = cam.zoom
        for x, y, vx, vy, lf, mlf, c, sz, spark in self.p:
            k = lf / mlf
            sx, sy = cam.w2s(x, y)
            if spark:
                col = tuple(int(v * k) for v in lighten(c, 0.5 * k))
                pg.draw.line(surf, col, (sx, sy), (sx - vx * 3 * zm, sy - vy * 3 * zm), max(2, int(2 * zm)))
            else:
                add_glow(surf, c, sx, sy, max(2, sz * 3 * zm * (0.4 + 0.6 * k)), intensity=k)


class Rings:
    def __init__(self):
        self.r = []

    def add(self, x, y, color, max_r=70):
        self.r.append([x, y, 4.0, max_r, color])

    def update(self):
        for q in self.r:
            q[2] += (q[3] - q[2]) * 0.1 + 0.6
        self.r = [q for q in self.r if q[2] < q[3] - 1]

    def draw(self, surf, cam):
        for x, y, rad, mr, c in self.r:
            k = 1 - rad / mr
            col = tuple(int(v * k) for v in c)
            sx, sy = cam.w2s(x, y)
            pg.draw.circle(surf, col, (int(sx), int(sy)), int(rad * cam.zoom), max(1, int(4 * k * cam.zoom)))


class FloatText:
    def __init__(self):
        self.t = []

    def add(self, s, x, y, color, size=20):
        self.t.append([s, x, y, 45, color, font(size * 0.6)])

    def update(self):
        for q in self.t:
            q[2] -= 0.7
            q[3] -= 1
        self.t = [q for q in self.t if q[3] > 0]

    def draw(self, surf, cam):
        for s, x, y, lf, c, f in self.t:
            text(surf, s, f, c, alpha=int(255 * min(1, lf / 20)), center=cam.w2s(x, y))


# ------------------------------------------------------------------ UI pieces (pixel rects)
def panel(surf, rect, border=None, alpha=200, radius=None, fill=(16, 18, 38)):
    radius = z(12) if radius is None else radius
    s = pg.Surface(rect.size, pg.SRCALPHA)
    pg.draw.rect(s, (*fill, alpha), s.get_rect(), border_radius=radius)
    hi = pg.Surface(rect.size, pg.SRCALPHA)
    pg.draw.rect(hi, (255, 255, 255, 12), (0, 0, rect.w, rect.h // 2), border_radius=radius)
    s.blit(hi, (0, 0))
    surf.blit(s, rect)
    if border:
        pg.draw.rect(surf, border, rect, 1, border_radius=radius)


_neon_cache = {}


def neon_rect(surf, rect, color, intensity=1.0, radius=None, width=2):
    radius = z(10) if radius is None else radius
    key = (rect.w, rect.h, color, radius, width)
    if key not in _neon_cache:
        if len(_neon_cache) > 200:
            _neon_cache.clear()
        g = pg.Surface((rect.w + 40, rect.h + 40), pg.SRCALPHA)
        pg.draw.rect(g, (*color, 255), (20, 20, rect.w, rect.h), width + 3, border_radius=radius)
        _neon_cache[key] = soft_blur(g, 5)
    g = _neon_cache[key]
    if intensity < 1:
        g = g.copy()
        g.set_alpha(int(255 * intensity))
    surf.blit(g, g.get_rect(center=rect.center), special_flags=pg.BLEND_RGB_ADD)
    pg.draw.rect(surf, lighten(color, 0.3), rect, width, border_radius=radius)


def seg_bar(surf, x, y, w, h, value, max_value, color, right_align=False, gap=None):
    gap = z(3) if gap is None else gap
    n = int(max_value)
    seg = (w - gap * (n - 1)) / n
    for i in range(n):
        idx = (n - 1 - i) if right_align else i
        on = idx < value
        rx = x + i * (seg + gap)
        if on:
            add_glow(surf, color, rx + seg / 2, y + h / 2, seg * 0.9, intensity=0.3)
        pg.draw.rect(surf, lighten(color, 0.15) if on else (34, 38, 64), (rx, y, seg, h), border_radius=z(3))


def keycap(surf, label, f, center, w=None, color=CYAN):
    tw = f.size(label)[0]
    r = pg.Rect(0, 0, w or max(z(40), tw + z(22)), z(36))
    r.center = center
    pg.draw.rect(surf, (12, 14, 30), r.move(0, z(3)), border_radius=z(8))
    pg.draw.rect(surf, (32, 36, 70), r, border_radius=z(8))
    pg.draw.rect(surf, darken(color, 0.3), r, max(1, z(1)), border_radius=z(8))
    text(surf, label, f, TEXT, shadow=False, center=r.center)
    return r


def fade(surf, alpha):
    if alpha <= 0:
        return
    s = pg.Surface(surf.get_size())
    s.fill((0, 0, 0))
    s.set_alpha(int(alpha))
    surf.blit(s, (0, 0))
