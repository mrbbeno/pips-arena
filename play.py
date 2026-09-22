"""Pip's Arena - the game (native 1920x1080, fullscreen by default).

    python play.py                     # main menu
    python play.py --mode duel --difficulty hard --map maze
    python play.py --mode pvp          # two players, one keyboard
    python play.py --mode survival

Controls: WASD / arrows = move, mouse = aim, left click / Space (hold) = shoot,
Esc / P = pause, Tab = AI debug overlay, M = mute, F11 = fullscreen.
Two players on one keyboard: P1 WASD + Space/F (L-Shift locks facing),
P2 arrows + Enter/R-Ctrl (R-Shift locks facing).

Testing flags: --bot strafer|rule (a bot plays for you), --frames N, --fast (no 60 fps cap),
--screenshot file.png, --shots-dir DIR --shots-every N, --mute, --seed, --windowed.
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pygame as pg

import config as C
import gfx as G
from bots import StraferAgent, rule_agent
from gfx import P, R, z
from maps import MAPS, random_map
from sound import Sound
from world import DIFFICULTIES, AIController, World, duel_view, pair_view

SAVE_PATH = os.environ.get("ARENA_SAVE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "save.json")
DIFF_KEYS = list(DIFFICULTIES)
DIFF_INFO = {  # tagline, strength 1..5
    "beginner": ("Lassan reagál és sokat hibázik. Bemelegítésnek.", 1),
    "easy": ("Már rád céloz, de kiszámítható és lassú.", 2),
    "medium": ("Kitér a lövéseid elől és elé céloz.", 3),
    "hard": ("Gyors reflexek, sarokba szorít, szórva lő.", 4),
    "impossible": ("A teljes erejű háló, késleltetés nélkül. Sok sikert.", 5),
}
MAP_KEYS = list(MAPS) + ["random"]
MAP_LABEL = {**{k: v["label"] for k, v in MAPS.items()}, "random": "Véletlen"}
P_NAMES = ["1. JÁTÉKOS", "2. JÁTÉKOS"]
WIN_ROUNDS = 3              # duel: first to 3 rounds
REPLAY_FRAMES = 90          # kill-cam replay: last 1.5 s ...
REPLAY_SPEED = 0.5          # ... at half speed
COUNTDOWN = 90
SURV_PLAYER_HP = 10
SURV_ENEMY_HP = 3
SURV_HEAL = 3
MAX_ENEMIES = 6
WAVES = [["beginner"], ["beginner", "beginner"], ["easy", "easy"], ["easy", "easy", "easy"],
         ["medium", "medium"], ["medium", "medium", "medium"], ["hard", "hard"],
         ["hard", "hard", "hard"], ["hard", "hard", "hard", "hard"], ["impossible", "hard", "hard", "hard"]]


def wave_enemies(k):
    if k <= len(WAVES):
        return WAVES[k - 1]
    n = min(4 + (k - len(WAVES)) // 2, MAX_ENEMIES)
    top = min(k - len(WAVES) + 1, n)
    return ["impossible"] * top + ["hard"] * (n - top)


def load_save():
    data = {"difficulty": "medium", "sound": True, "volume": 0.7, "shake": True, "debug": False,
            "fullscreen": True, "map": "pillars", "items": True,
            "best_survival": 0, "best_wave": 0, "duel": {}, "pvp": [0, 0]}
    try:
        with open(SAVE_PATH, encoding="utf-8") as f:
            data.update(json.load(f))
    except (OSError, ValueError):
        pass
    if data["difficulty"] not in DIFFICULTIES:
        data["difficulty"] = "medium"
    if data["map"] not in MAP_KEYS:
        data["map"] = "pillars"
    return data


def new_stats():
    return {"shots": [0, 0], "hits": [0, 0], "dodges": [0, 0], "frames": 0}


def acc(st, k):
    return st["hits"][k] / st["shots"][k] if st["shots"][k] else 0.0


def ease_out(t):
    return 1 - (1 - min(max(t, 0), 1)) ** 3


def full_map_camera():
    """Camera showing the whole map below the HUD bar (two-player mode, menu background)."""
    cam = G.Camera(zoom=min(G.SW / C.W, (G.SH - G.HUD_H) / C.H))
    cam.cx = C.W / 2
    cam.cy = C.H / 2 - G.HUD_H / 2 / cam.zoom
    return cam


class Game:
    def __init__(self, args):
        self.args = args
        pg.init()
        pg.display.set_caption("Pip's Arena")
        self.save = load_save()
        if args.difficulty:
            self.save["difficulty"] = args.difficulty
        if args.map:
            self.save["map"] = args.map
        if args.windowed:
            self.save["fullscreen"] = False
        self.screen = self._set_display(self.save["fullscreen"])
        self.clock = pg.time.Clock()
        self.f_hero = G.font(84)
        self.f_title = G.font(46)
        self.f_big = G.font(38)
        self.f_mid = G.font(23)
        self.f_btn = G.font(20)
        self.f_small = G.font(15, bold=False)
        self.f_tiny = G.font(12.5, bold=False)
        self.f_mono = G.mono(12)
        self.sound = Sound(enabled=self.save["sound"] and not args.mute, volume=self.save["volume"])
        self.particles, self.rings, self.ftext = G.Particles(), G.Rings(), G.FloatText()
        self.ui_particles = G.Particles()
        self.screen_cam = G.ScreenCam()
        self.cam = full_map_camera()
        self.cam.shake_on = self.save["shake"]
        self.flash = np.zeros(1 + MAX_ENEMIES)
        self.hurt_flash = 0
        self.freeze = 0
        self.muzzles = []            # [x, y, color, life] world-space muzzle flashes
        self.dmg_dirs = []           # [angle, life] directional damage indicators
        self.trails = [deque(maxlen=7) for _ in range(1 + MAX_ENEMIES)]
        self.cross_spread = 0.0
        self.debug = self.save["debug"]
        self.frame = 0
        self.running = True
        self.bot = {"strafer": StraferAgent(seed=args.seed), "rule": rule_agent}.get(args.bot)
        self.bot2 = None
        self.infer_us = deque(maxlen=120)
        self.last_dodge_txt = -999
        self.focus = None
        self.pressed = None
        self.anim = {}
        self.fade_in = 0
        self.scene = None
        self.menu_shade = self._menu_shade()
        self.demo = None
        self.pvp = False
        self.next_mode = "duel"
        self.facing = [0.0, math.pi]
        self._prewarm()
        # GC: move everything loaded so far out of the collector's way and make collections rare;
        # a full collect happens on every scene change (between rounds/menus) instead of mid-fight.
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 50, 50)
        self.goto(args.mode)

    def _prewarm(self):
        """Render the big glowing banner texts and load every AI model up front, so nothing is
        generated in the middle of a match (those were the in-game hitches)."""
        tmp = pg.Surface((10, 10))
        titles = ["HARC!", "KÖRT NYERTED", "PIP NYERTE A KÖRT", "DÖNTETLEN", "HULLÁM TELJESÍTVE",
                  f"{P_NAMES[0]} VISZI A KÖRT", f"{P_NAMES[1]} VISZI A KÖRT"] + [f"{i}. HULLÁM" for i in range(1, 21)]
        for t in titles:
            for col, glow in ((G.GOLD, None), (G.lighten(G.CYAN, 0.55), G.CYAN), (G.lighten(G.GOLD, 0.55), G.GOLD),
                              (G.lighten(G.TEXT, 0.55), G.TEXT), (G.lighten(G.PINK, 0.55), G.PINK),
                              (G.lighten(G.DIM, 0.55), G.DIM)):
                f = self.f_title if t == "HARC!" else self.f_big
                G.glow_text(tmp, t, f, col, glow=glow, spacing=8 if t == "HARC!" else 4, center=(0, 0))
        for key in DIFFICULTIES:
            AIController.from_difficulty(key)

    # ================================================================ plumbing
    def _set_display(self, fullscreen):
        flags = pg.SCALED | (pg.FULLSCREEN if fullscreen else pg.RESIZABLE)
        try:
            return pg.display.set_mode((G.SW, G.SH), flags, vsync=1)
        except pg.error:
            return pg.display.set_mode((G.SW, G.SH), flags)

    def toggle_fullscreen(self):
        try:
            pg.display.toggle_fullscreen()
        except pg.error:
            pass
        self.save["fullscreen"] = not self.save["fullscreen"]
        self.write_save()

    def goto(self, scene):
        gc.collect(1)
        self.scene = scene
        self.fade_in = 255
        self.pressed = None
        pg.mouse.set_visible(scene not in ("duel", "survival", "pvp"))
        if scene in ("menu", "select", "settings", "controls", "maps") and self.demo is None:
            self._setup_demo()
        if scene == "menu":
            self.focus = "duel"
        elif scene == "select":
            self.focus = self.save["difficulty"]
        elif scene == "maps":
            self.focus = self.save["map"]
        elif scene == "settings":
            self.focus = "volume"
        elif scene == "controls":
            self.focus = "back"
        elif scene in ("duel", "pvp"):
            self.demo = None
            self.start_duel(pvp=scene == "pvp")
        elif scene == "survival":
            self.demo = None
            self.start_survival()
        elif scene == "results":
            self.focus = "again"

    def run(self):
        while self.running:
            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT:
                    self.running = False
                elif e.type == pg.KEYDOWN and e.key == pg.K_F11:
                    self.toggle_fullscreen()
                elif e.type == pg.KEYDOWN and e.key == pg.K_m:
                    self.sound.enabled = not self.sound.enabled
                    self.save["sound"] = self.sound.enabled
                    self.write_save()
            getattr(self, "update_" + self.scene)(events)
            getattr(self, "draw_" + self.scene)()
            if self.fade_in > 0:
                G.fade(self.screen, self.fade_in)
                self.fade_in = max(0, self.fade_in - 28)
            pg.display.flip()
            self.clock.tick(0 if self.args.fast else C.FPS)
            self.frame += 1
            a = self.args
            if a.shots_dir and a.shots_every and self.frame % a.shots_every == 0:
                os.makedirs(a.shots_dir, exist_ok=True)
                pg.image.save(self.screen, os.path.join(a.shots_dir, f"{self.frame:05d}_{self.scene}.png"))
            if a.frames and self.frame >= a.frames:
                self.running = False
        if self.args.screenshot:
            pg.image.save(self.screen, self.args.screenshot)
        self.write_save()
        pg.quit()

    def write_save(self):
        try:
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.save, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    # ================================================================ UI core
    def nav(self, events, items, axis="y", back=None, cols=None):
        """Shared mouse + keyboard handling for a list of (id, rect) widgets.
        Mouse: hover focuses, activation on button *release* over the same widget.
        Keys: arrows/WASD move focus (grid when `cols`), Enter/Space activate, Esc -> `back`."""
        ids = [i for i, _ in items]
        if self.focus not in ids and ids:
            self.focus = ids[0]
        prev_k = (pg.K_UP, pg.K_w) if axis == "y" else (pg.K_LEFT, pg.K_a)
        next_k = (pg.K_DOWN, pg.K_s) if axis == "y" else (pg.K_RIGHT, pg.K_d)
        hovered = next((i for i, r in items if r.collidepoint(pg.mouse.get_pos())), None)
        for e in events:
            if e.type == pg.MOUSEMOTION and hovered is not None and hovered != self.focus:
                self.focus = hovered
                self.sound.play("hover")
            elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                self.pressed = hovered
            elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
                hit = hovered if hovered is not None and hovered == self.pressed else None
                self.pressed = None
                if hit is not None:
                    self.focus = hit
                    return hit
            elif e.type == pg.KEYDOWN:
                grid_keys = (pg.K_LEFT, pg.K_a, pg.K_RIGHT, pg.K_d, pg.K_UP, pg.K_w, pg.K_DOWN, pg.K_s)
                if cols and e.key in grid_keys:
                    k = ids.index(self.focus)
                    step = {pg.K_LEFT: -1, pg.K_a: -1, pg.K_RIGHT: 1, pg.K_d: 1,
                            pg.K_UP: -cols, pg.K_w: -cols, pg.K_DOWN: cols, pg.K_s: cols}[e.key]
                    self.focus = ids[min(max(k + step, 0), len(ids) - 1)]
                    self.sound.play("hover")
                elif e.key in prev_k + next_k and ids:
                    k = ids.index(self.focus)
                    self.focus = ids[(k + (-1 if e.key in prev_k else 1)) % len(ids)]
                    self.sound.play("hover")
                elif e.key in (pg.K_RETURN, pg.K_SPACE, pg.K_KP_ENTER):
                    return self.focus
                elif e.key in (pg.K_ESCAPE, pg.K_BACKSPACE) and back:
                    return back
        return None

    def a(self, key, target, speed=0.22):
        v = self.anim.get(key, 0.0)
        v += (target - v) * speed
        self.anim[key] = v
        return v

    def button(self, id_, rect, label, color=G.CYAN, align="left", sub=None):
        t = self.a(id_, 1.0 if self.focus == id_ else 0.0)
        s = self.screen
        r = rect.move(0, z(1) if self.pressed == id_ else 0)
        G.panel(s, r, alpha=int(90 + 110 * t), fill=G.mix((14, 16, 34), G.darken(color, 0.72), t), radius=z(9))
        if t > 0.02:
            G.neon_rect(s, r, color, intensity=0.55 * t, radius=z(9), width=2)
            bar = pg.Rect(r.x, r.y + r.h * (0.5 - 0.35 * t), z(3), r.h * 0.7 * t)
            pg.draw.rect(s, color, bar, border_radius=z(2))
        col = G.mix(G.DIM, G.TEXT, t)
        if align == "left":
            G.text(s, label, self.f_btn, col, spacing=2, midleft=(r.x + z(20) + z(9) * t, r.centery))
            if t > 0.05:
                G.text(s, "›", self.f_mid, G.mix(G.BG1, color, t), shadow=False, midright=(r.right - z(14), r.centery - z(1)))
        else:
            G.text(s, label, self.f_btn, col, spacing=2, center=r.center)
        if sub:
            G.text(s, sub, self.f_tiny, G.FAINT, shadow=False, midright=(r.right - z(34), r.centery))

    def _menu_shade(self):
        s = pg.Surface((G.SW, G.SH), pg.SRCALPHA)
        x = np.arange(G.SW)
        a = np.clip(245 - x / z(560) * 190, 55, 245).astype(np.uint8)
        s.fill((6, 7, 15, 0))
        pg.surfarray.pixels_alpha(s)[:] = np.repeat(a[:, None], G.SH, 1)
        return s

    def dim(self, alpha=190):
        sh = pg.Surface((G.SW, G.SH), pg.SRCALPHA)
        sh.fill((5, 6, 14, alpha))
        self.screen.blit(sh, (0, 0))

    def back_button_items(self):
        return [("back", R(28, 540 - 62, 150, 42))]

    # ---------------------------------------------------------------- menu background: live AI vs AI
    def _setup_demo(self):
        self.demo = World([C.MAX_HP, C.MAX_HP], team=[0, 1], obstacles=MAPS["pillars"]["rects"], items=True)
        self.demo.reset_duel()
        self.demo_ai = [AIController.from_difficulty("impossible"), AIController.from_difficulty("hard")]
        self.demo_colors = [G.CYAN, G.ENEMY_C["hard"]]
        self.demo_cam = full_map_camera()
        self.clear_fx()

    def update_demo(self):
        w = self.demo
        views = [pair_view(w, [0], 1, t_cap=10 ** 9), pair_view(w, [1], 0, t_cap=10 ** 9)]
        acts = [ai.act(v)[:3] for ai, v in zip(self.demo_ai, views)]
        ev = w.step(np.stack([a_[0][0] for a_ in acts]), np.array([a_[1][0] for a_ in acts]),
                    np.array([a_[2][0] for a_ in acts]))
        for s, t, x, y, _absorbed in ev.hits:
            self.flash[t] = 6
            self.particles.burst(x, y, self.demo_colors[t], n=10, speed=5, life=18, spark=True)
        for i, x, y in ev.deaths:
            self.particles.burst(x, y, self.demo_colors[i], n=40, speed=6, life=40, size=4)
            self.rings.add(x, y, self.demo_colors[i], 90)
        for s, x, y in ev.impacts:
            self.particles.burst(x, y, self.demo_colors[s], n=5, speed=3, life=10, spark=True)
        if not w.alive.all() or w.t >= C.MAX_TICKS:
            w.reset_duel()
        self.update_fx()

    def draw_demo_bg(self, shade=True):
        s = self.screen
        G.draw_world(s, self.demo, self.demo_colors, self.demo_cam, self.flash, t=self.frame)
        self.rings.draw(s, self.demo_cam)
        self.particles.draw(s, self.demo_cam)
        if shade:
            s.blit(self.menu_shade, (0, 0))

    # ================================================================ menu
    MENU = [("duel", "PÁRBAJ PIP ELLEN"), ("pvp", "KÉT JÁTÉKOS"), ("survival", "TÚLÉLÉS"),
            ("settings", "BEÁLLÍTÁSOK"), ("controls", "IRÁNYÍTÁS"), ("quit", "KILÉPÉS")]

    def menu_items(self):
        return [(k, R(56, 226 + i * 44, 420, 40)) for i, (k, _) in enumerate(self.MENU)]

    def update_menu(self, events):
        self.update_demo()
        hit = self.nav(events, self.menu_items(), back="quit")
        if hit == "quit":
            self.running = False
        elif hit == "duel":
            self.sound.play("select")
            self.goto("select")
        elif hit in ("pvp", "survival"):
            self.sound.play("select")
            self.next_mode = hit
            self.goto("maps")
        elif hit:
            self.sound.play("select")
            self.goto(hit)

    def draw_menu(self):
        s = self.screen
        self.draw_demo_bg()
        k = ease_out(1 - self.fade_in / 255)
        G.glow_text(s, "PIP'S", self.f_hero, G.PINK, spacing=6, topleft=P(50 - 30 * (1 - k), 26))
        G.glow_text(s, "ARENA", self.f_hero, G.CYAN, spacing=6, topleft=P(50 - 60 * (1 - k), 104))
        G.text(s, "Pip egy önmaga ellen tanult neurális háló. Győzd le.", self.f_small, G.DIM, topleft=P(56, 196))
        labels = dict(self.MENU)
        rec = self.save["duel"]
        wins = sum(v[0] for v in rec.values())
        losses = sum(v[1] for v in rec.values())
        pv = self.save["pvp"]
        subs = {"duel": f"{wins} – {losses}" if wins + losses else None,
                "pvp": f"{pv[0]} – {pv[1]}" if sum(pv) else None,
                "survival": f"rekord {self.save['best_survival']}" if self.save["best_survival"] else None}
        for kid, r in self.menu_items():
            self.button(kid, r, labels[kid], color=G.PINK if kid == "quit" else G.CYAN, sub=subs.get(kid))
        on = (self.frame // 30) % 2 == 0
        tag = R(960 - 190, 18, 166, 28)
        G.panel(s, tag, border=G.FAINT, alpha=170, radius=tag.h // 2)
        pg.draw.circle(s, (255, 60, 80) if on else (110, 30, 40), (tag.x + z(16), tag.centery), z(4.5))
        G.text(s, "ÉLŐ  ·  PIP vs PIP", self.f_tiny, G.TEXT, shadow=False, midleft=(tag.x + z(30), tag.centery))
        G.text(s, "nyilak / egér   ·   Enter / klikk   ·   F11: teljes képernyő", self.f_tiny, G.FAINT,
               shadow=False, topleft=P(56, 540 - 24))

    # ================================================================ difficulty select
    def select_items(self):
        w, h, gap = 162, 300, 14
        x0 = (960 - (5 * w + 4 * gap)) // 2
        items = [(k, R(x0 + i * (w + gap), 126, w, h)) for i, k in enumerate(DIFF_KEYS)]
        return items + self.back_button_items()

    def update_select(self, events):
        self.update_demo()
        hit = self.nav(events, self.select_items(), axis="x", back="back")
        for e in events:  # Down -> back button, Up -> back to the cards
            if e.type == pg.KEYDOWN and e.key in (pg.K_DOWN, pg.K_s) and self.focus != "back":
                self._last_card, self.focus = self.focus, "back"
                self.sound.play("hover")
            elif e.type == pg.KEYDOWN and e.key in (pg.K_UP, pg.K_w) and self.focus == "back":
                self.focus = getattr(self, "_last_card", self.save["difficulty"])
                self.sound.play("hover")
        if hit == "back":
            self.sound.play("back")
            self.goto("menu")
        elif hit in DIFFICULTIES:
            self.save["difficulty"] = hit
            self.write_save()
            self.sound.play("select")
            self.next_mode = "duel"
            self.goto("maps")

    def draw_select(self):
        s = self.screen
        self.draw_demo_bg(shade=False)
        self.dim(205)
        G.glow_text(s, "VÁLASSZ ELLENFELET", self.f_title, G.TEXT, glow=G.CYAN, spacing=3, center=P(480, 52))
        G.text(s, "Párbaj  ·  3 nyert körig  ·  minden szint ugyanaz a háló, más reakcióidővel",
               self.f_small, G.DIM, center=P(480, 94))
        for key, r in self.select_items()[:5]:
            d = DIFFICULTIES[key]
            col = G.ENEMY_C[key]
            t = self.a("card_" + key, 1.0 if self.focus == key else 0.0)
            rr = r.move(0, -z(10) * t)
            G.panel(s, rr, alpha=int(150 + 80 * t), fill=G.mix((13, 15, 32), G.darken(col, 0.8), 0.5 + 0.5 * t),
                    radius=z(14))
            if t > 0.02:
                G.neon_rect(s, rr, col, intensity=t, radius=z(14), width=2)
            else:
                pg.draw.rect(s, G.FAINT, rr, 1, border_radius=z(14))
            cx = rr.centerx
            oy = rr.y + z(64)
            pulse = 0.8 + 0.2 * math.sin(self.frame * 0.08 + DIFF_KEYS.index(key))
            G.add_glow(s, col, cx, oy, z(56), intensity=(0.45 + 0.5 * t) * pulse)
            pg.draw.circle(s, G.darken(col, 0.5), (cx, oy), z(21))
            pg.draw.circle(s, G.lighten(col, 0.3), (cx, oy), z(21), z(3))
            pg.draw.circle(s, G.lighten(col, 0.7), (cx, oy), z(6))
            G.text(s, d["label"].upper(), self.f_btn, G.lighten(col, 0.2 * t), spacing=1, center=(cx, rr.y + z(120)))
            reac = f"{d['delay'] * 1000 // C.FPS} ms reakció" if d["delay"] else "azonnali reakció"
            G.text(s, reac, self.f_tiny, G.DIM, shadow=False, center=(cx, rr.y + z(144)))
            level = DIFF_INFO[key][1]
            for i in range(5):
                x = cx + (i - 2) * z(18) - z(7)
                on = i < level
                if on:
                    G.add_glow(s, col, x + z(7), rr.y + z(168), z(12), intensity=0.5)
                pg.draw.rect(s, G.lighten(col, 0.1) if on else (36, 40, 70), (x, rr.y + z(164), z(14), z(8)),
                             border_radius=z(3))
            for j, line in enumerate(G.wrap(DIFF_INFO[key][0], self.f_tiny, r.w - z(22))):
                G.text(s, line, self.f_tiny, G.mix(G.DIM, G.TEXT, t), shadow=False, center=(cx, rr.y + z(198 + j * 17)))
            rec = self.save["duel"].get(key)
            G.text(s, f"{rec[0]} – {rec[1]}" if rec else "–", self.f_mid, G.TEXT if rec else G.FAINT,
                   center=(cx, rr.bottom - z(38)))
            G.text(s, "mérleg", self.f_tiny, G.FAINT, shadow=False, center=(cx, rr.bottom - z(16)))
        self.button("back", self.back_button_items()[0][1], "VISSZA", color=G.PINK, align="center")
        G.text(s, "nyilak: választás   ·   Enter / klikk: tovább   ·   Esc: vissza", self.f_tiny, G.FAINT,
               shadow=False, midright=P(960 - 28, 540 - 40))

    # ================================================================ map select
    def maps_items(self):
        w, h, gap = 196, 146, 16
        x0 = (960 - (4 * w + 3 * gap)) // 2
        items = [(k, R(x0 + (i % 4) * (w + gap), 118 + (i // 4) * (h + gap), w, h)) for i, k in enumerate(MAP_KEYS)]
        items.append(("items", R(x0 + 3 * (w + gap), 118 + (h + gap), w, h)))
        return items + self.back_button_items()

    def update_maps(self, events):
        self.update_demo()
        hit = self.nav(events, self.maps_items(), back="back", cols=4)
        if hit == "back":
            self.sound.play("back")
            self.goto("select" if self.next_mode == "duel" else "menu")
        elif hit == "items":
            self.save["items"] = not self.save["items"]
            self.write_save()
            self.sound.play("select")
        elif hit in MAP_KEYS:
            self.save["map"] = hit
            self.write_save()
            self.sound.play("select")
            self.goto(self.next_mode)

    def draw_maps(self):
        s = self.screen
        self.draw_demo_bg(shade=False)
        self.dim(205)
        G.glow_text(s, "VÁLASSZ PÁLYÁT", self.f_title, G.TEXT, glow=G.CYAN, spacing=3, center=P(480, 46))
        mode = {"duel": f"Párbaj · Pip ({DIFFICULTIES[self.save['difficulty']]['label']})", "pvp": "Két játékos",
                "survival": "Túlélés"}[self.next_mode]
        G.text(s, f"{mode}  ·  a fedezék elnyeli a lövéseket", self.f_small, G.DIM, center=P(480, 86))
        if not hasattr(self, "_map_prev"):
            rng = np.random.default_rng(7)
            self._map_prev = {k: G.map_preview(MAPS[k]["rects"] if k in MAPS else random_map(rng), z(176), z(99))
                              for k in MAP_KEYS}
        for key, r in self.maps_items()[:-1]:
            t = self.a("map_" + key, 1.0 if self.focus == key else 0.0)
            rr = r.move(0, -z(6) * t)
            col = G.GOLD if key == "items" else G.CYAN
            G.panel(s, rr, alpha=int(150 + 80 * t), fill=G.mix((13, 15, 32), G.darken(col, 0.8), 0.4 + 0.6 * t),
                    radius=z(14))
            if t > 0.02:
                G.neon_rect(s, rr, col, intensity=t, radius=z(14), width=2)
            else:
                pg.draw.rect(s, G.FAINT, rr, 1, border_radius=z(14))
            if key == "items":
                on = self.save["items"]
                for i in range(5):
                    G.draw_item_icon(s, i, rr.centerx + (i - 2) * z(32), rr.y + z(45), z(11), self.frame, glow_on=on)
                G.text(s, "TÁRGYAK", self.f_btn, G.TEXT, spacing=2, center=(rr.centerx, rr.y + z(90)))
                G.text(s, "BE" if on else "KI", self.f_mid, G.GOLD if on else G.DIM, center=(rr.centerx, rr.y + z(120)))
                continue
            prev = self._map_prev[key]
            s.blit(prev, prev.get_rect(midtop=(rr.centerx, rr.y + z(10))))
            if key == "random":
                G.text(s, "?", self.f_title, G.lighten(G.CYAN, 0.4), center=(rr.centerx, rr.y + z(58)))
            G.text(s, MAP_LABEL[key].upper(), self.f_small, G.mix(G.DIM, G.TEXT, t), spacing=1,
                   center=(rr.centerx, rr.bottom - z(20)))
        self.button("back", self.back_button_items()[0][1], "VISSZA", color=G.PINK, align="center")
        G.text(s, "nyilak: választás   ·   Enter / klikk: indítás   ·   Esc: vissza", self.f_tiny, G.FAINT,
               shadow=False, midright=P(960 - 28, 540 - 40))

    def current_map(self):
        key = self.save["map"]
        return random_map(np.random.default_rng()) if key == "random" else MAPS[key]["rects"]

    # ================================================================ settings
    SETTINGS = [("volume", "HANGERŐ"), ("fullscreen", "TELJES KÉPERNYŐ"),
                ("shake", "KÉPERNYŐRÁZÁS"), ("debug", "AI GONDOLATAI")]

    def settings_items(self):
        return [(k, R(230, 116 + i * 58, 500, 50)) for i, (k, _) in enumerate(self.SETTINGS)] + self.back_button_items()

    def slider_track(self, key):
        r = dict(self.settings_items())[key]
        return pg.Rect(r.x + z(230), r.centery - z(4), z(190), z(8))

    def update_settings(self, events):
        self.update_demo()
        hit = self.nav(events, self.settings_items(), back="back")
        for e in events:
            if e.type == pg.KEYDOWN and self.focus == "volume" and \
                    e.key in (pg.K_LEFT, pg.K_a, pg.K_RIGHT, pg.K_d):
                self.set_level(self.focus, self.save[self.focus] + (0.1 if e.key in (pg.K_RIGHT, pg.K_d) else -0.1))
            if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                for key in ("volume",):
                    tr = self.slider_track(key)
                    if tr.inflate(z(30), z(30)).collidepoint(e.pos):
                        self.dragging = key
                        self.set_level(key, round((e.pos[0] - tr.x) / tr.w * 20) / 20)
            elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
                self.dragging = None
        drag = getattr(self, "dragging", None)
        if drag and pg.mouse.get_pressed()[0]:
            tr = self.slider_track(drag)
            v = round((pg.mouse.get_pos()[0] - tr.x) / tr.w * 20) / 20
            if abs(np.clip(v, 0, 1) - self.save[drag]) > 1e-6:
                self.set_level(drag, v)
        if hit == "back":
            self.sound.play("back")
            self.goto("menu")
        elif hit == "shake":
            self.save["shake"] = self.cam.shake_on = not self.save["shake"]
            self.sound.play("select")
        elif hit == "debug":
            self.save["debug"] = self.debug = not self.save["debug"]
            self.sound.play("select")
        elif hit == "fullscreen":
            self.toggle_fullscreen()
            self.sound.play("select")
        if hit:
            self.write_save()

    def set_level(self, key, v):
        v = float(np.clip(round(v, 2), 0, 1))
        self.save[key] = v
        self.sound.volume = v
        self.sound.enabled = v > 0
        self.save["sound"] = v > 0
        self.sound.play("hit")
        self.write_save()

    def draw_settings(self):
        s = self.screen
        self.draw_demo_bg(shade=False)
        self.dim(205)
        G.glow_text(s, "BEÁLLÍTÁSOK", self.f_title, G.TEXT, glow=G.CYAN, spacing=3, center=P(480, 62))
        items = dict(self.settings_items())
        for key, label in self.SETTINGS:
            r = items[key]
            self.button(key, r, label)
            if key == "volume":
                tr = self.slider_track(key)
                v = self.save[key]
                pg.draw.rect(s, (36, 40, 70), tr, border_radius=z(4))
                fill = tr.copy()
                fill.w = int(tr.w * v)
                pg.draw.rect(s, G.CYAN, fill, border_radius=z(4))
                G.add_glow(s, G.CYAN, tr.x + fill.w, tr.centery, z(18), intensity=0.8)
                pg.draw.circle(s, G.TEXT, (tr.x + fill.w, tr.centery), z(8))
                G.text(s, f"{int(v * 100)}%", self.f_btn, G.TEXT, midleft=(tr.right + z(16), tr.centery))
            else:
                on = self.save[key]
                sw = pg.Rect(r.right - z(84), r.centery - z(13), z(52), z(26))
                pg.draw.rect(s, G.darken(G.CYAN, 0.4) if on else (36, 40, 70), sw, border_radius=z(13))
                kx = sw.right - z(13) if on else sw.x + z(13)
                if on:
                    G.add_glow(s, G.CYAN, kx, sw.centery, z(20), intensity=0.7)
                pg.draw.circle(s, G.TEXT, (kx, sw.centery), z(10))
                G.text(s, "BE" if on else "KI", self.f_tiny, G.TEXT if on else G.DIM, shadow=False,
                       midright=(sw.x - z(12), sw.centery))
        G.text(s, "M: hang be/ki   ·   F11: teljes képernyő   ·   Tab: AI gondolatai játék közben", self.f_tiny,
               G.FAINT, shadow=False, center=P(480, 420))
        self.button("back", items["back"], "VISSZA", color=G.PINK, align="center")

    # ================================================================ controls
    def update_controls(self, events):
        self.update_demo()
        hit = self.nav(events, self.back_button_items(), back="back")
        if hit == "back":
            self.sound.play("back")
            self.goto("menu")

    def draw_controls(self):
        s = self.screen
        self.draw_demo_bg(shade=False)
        self.dim(210)
        G.glow_text(s, "IRÁNYÍTÁS", self.f_title, G.TEXT, glow=G.CYAN, spacing=3, center=P(480, 44))
        f, fs = self.f_btn, self.f_small

        def wasd(cx, cy, keys, color):
            G.keycap(s, keys[0], f, (cx, cy - z(22)), z(44), color)
            for i, k in enumerate(keys[1:]):
                G.keycap(s, k, f, (cx - z(48) + i * z(48), cy + z(22)), z(44), color)

        left = R(40, 84, 400, 300)
        G.panel(s, left, border=G.FAINT, alpha=150)
        G.text(s, "EGY JÁTÉKOS", f, G.CYAN, spacing=2, center=(left.centerx, left.y + z(26)))
        wasd(left.x + z(105), left.y + z(110), "WASD", G.CYAN)
        G.text(s, "mozgás", fs, G.TEXT, center=(left.x + z(105), left.y + z(160)))
        mx, my = left.x + z(290), left.y + z(105)
        body = pg.Rect(0, 0, z(54), z(80))
        body.center = (mx, my)
        pg.draw.rect(s, (32, 36, 70), body, border_radius=z(27))
        pg.draw.rect(s, G.darken(G.CYAN, 0.3), body, 1, border_radius=z(27))
        pg.draw.rect(s, G.darken(G.PINK, 0.35), (body.x, body.y, z(27), z(34)), border_top_left_radius=z(27))
        G.text(s, "célzás + lövés", fs, G.TEXT, center=(mx, left.y + z(160)))
        G.keycap(s, "SPACE", f, (left.centerx, left.y + z(220)), z(150), G.CYAN)
        G.text(s, "lövés billentyűvel", fs, G.DIM, center=(left.centerx, left.y + z(256)))
        G.text(s, "Párbaj Pip ellen és Túlélés", self.f_tiny, G.FAINT, shadow=False,
               center=(left.centerx, left.bottom - z(18)))
        right = R(960 - 40 - 440, 84, 440, 300)
        G.panel(s, right, border=G.FAINT, alpha=150)
        G.text(s, "KÉT JÁTÉKOS", f, G.PINK, spacing=2, center=(right.centerx, right.y + z(26)))
        c1, c2 = right.x + z(105), right.x + z(335)
        G.text(s, "1. JÁTÉKOS", fs, G.CYAN, center=(c1, right.y + z(62)))
        G.text(s, "2. JÁTÉKOS", fs, G.PINK, center=(c2, right.y + z(62)))
        wasd(c1, right.y + z(120), "WASD", G.CYAN)
        wasd(c2, right.y + z(120), "↑←↓→" if f.size("↑")[0] > z(4) else "^<v>", G.PINK)
        G.keycap(s, "SPACE / F", self.f_small, (c1, right.y + z(196)), z(130), G.CYAN)
        G.keycap(s, "ENTER / R-CTRL", self.f_small, (c2, right.y + z(196)), z(150), G.PINK)
        G.text(s, "lövés", fs, G.DIM, center=(right.centerx, right.y + z(196)))
        G.keycap(s, "L-SHIFT", self.f_small, (c1, right.y + z(246)), z(110), G.CYAN)
        G.keycap(s, "R-SHIFT", self.f_small, (c2, right.y + z(246)), z(110), G.PINK)
        G.text(s, "irány", fs, G.DIM, center=(right.centerx, right.y + z(238)))
        G.text(s, "rögzítés", fs, G.DIM, center=(right.centerx, right.y + z(256)))
        G.text(s, "Arra lősz, amerre utoljára mentél. Shifttel oldalazva is tartod az irányt.", self.f_tiny,
               G.FAINT, shadow=False, center=(right.centerx, right.bottom - z(18)))
        rows = [("ESC", "szünet"), ("TAB", "AI gondolatai"), ("M", "hang"), ("F11", "teljes képernyő")]
        for i, (k, v) in enumerate(rows):
            x = z(110 + i * 215)
            G.keycap(s, k, self.f_small, (x, z(418)), z(56))
            G.text(s, v, fs, G.TEXT, midleft=(x + z(36), z(418)))
        G.text(s, "Tárgyak: sörétes · gépfegyver · railgun (2 sebzés) · pajzs · élet — rámenve felveszed.",
               self.f_small, G.DIM, midleft=P(210, 540 - 41))
        self.button("back", self.back_button_items()[0][1], "VISSZA", color=G.PINK, align="center")

    # ================================================================ shared gameplay
    def clear_fx(self):
        self.particles, self.rings, self.ftext = G.Particles(), G.Rings(), G.FloatText()
        self.flash[:] = 0
        self.hurt_flash = 0
        self.muzzles, self.dmg_dirs = [], []
        for tr in self.trails:
            tr.clear()

    def update_fx(self, slow=False):
        if slow and self.frame % 2:
            return
        self.particles.update()
        self.rings.update()
        self.ftext.update()
        self.flash = np.maximum(self.flash - 1, 0)
        self.hurt_flash = max(self.hurt_flash - 1, 0)
        for m in self.muzzles:
            m[3] -= 1
        self.muzzles = [m for m in self.muzzles if m[3] > 0]
        for d in self.dmg_dirs:
            d[1] -= 1
        self.dmg_dirs = [d for d in self.dmg_dirs if d[1] > 0]
        self.cross_spread *= 0.85

    def update_trails(self, w):
        for k in range(w.N):
            tr = self.trails[k]
            if w.alive[k] and np.hypot(*w.vel[k]) > 1.5:
                tr.append(tuple(w.pos[k]))
            elif tr:
                tr.popleft()

    P_KEYS = [dict(up=(pg.K_w,), down=(pg.K_s,), left=(pg.K_a,), right=(pg.K_d,), shoot=(pg.K_SPACE, pg.K_f),
                   lock=(pg.K_LSHIFT,)),
              dict(up=(pg.K_UP,), down=(pg.K_DOWN,), left=(pg.K_LEFT,), right=(pg.K_RIGHT,),
                   shoot=(pg.K_RETURN, pg.K_RCTRL, pg.K_KP0, pg.K_KP_ENTER), lock=(pg.K_RSHIFT,))]

    def keyboard_player(self, w, i):
        """Two-players-one-keyboard input: shoot where you last moved (Shift keeps the facing)."""
        if self.bot:
            bot = self.bot if i == 0 else self.bot2
            m, a, s = bot(pair_view(w, [i], 1 - i, t_cap=10 ** 9), 0)
            return np.asarray(m, dtype=float)[0], float(np.asarray(a)[0]), bool(np.asarray(s)[0])
        keys = pg.key.get_pressed()
        k = self.P_KEYS[i]
        down = lambda names: any(keys[c] for c in names)
        dx = down(k["right"]) - down(k["left"])
        dy = down(k["down"]) - down(k["up"])
        if (dx or dy) and not down(k["lock"]):
            self.facing[i] = math.atan2(dy, dx)
        return np.array([dx, dy], dtype=float), self.facing[i], down(k["shoot"])

    def player_input(self, w):
        if self.bot:
            enemies = np.flatnonzero(w.alive[1:]) + 1
            if not enemies.size:
                return np.zeros(2), w.aim[0], False
            tgt = enemies[np.argmin(np.linalg.norm(w.pos[enemies] - w.pos[0], axis=-1))]
            m, a, s = self.bot(pair_view(w, [0], tgt, t_cap=10 ** 9), 0)
            return np.asarray(m, dtype=float)[0], float(np.asarray(a)[0]), bool(np.asarray(s)[0])
        keys = pg.key.get_pressed()
        dx = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
        dy = (keys[pg.K_s] or keys[pg.K_DOWN]) - (keys[pg.K_w] or keys[pg.K_UP])
        mx, my = self.cam.s2w(*pg.mouse.get_pos())
        aim = math.atan2(my - w.pos[0, 1], mx - w.pos[0, 0])
        shoot = keys[pg.K_SPACE] or pg.mouse.get_pressed()[0]
        return np.array([dx, dy], dtype=float), aim, bool(shoot)

    def handle_events(self, ev, colors, stats):
        w = self.w
        shot_sound = set()
        for i, x, y, a, wpn in ev.fired:
            self.muzzles.append([x, y, colors[i], 4])
            self.particles.burst(x, y, colors[i], n=5, speed=3, life=9, size=2, spread=0.7, angle=a, spark=True)
            if i in shot_sound:        # one sound/kick per trigger pull (the shotgun fires 3 pellets)
                continue
            shot_sound.add(i)
            stats["shots"][min(i, 1)] += 1
            name = {1: "shoot_shotgun", 3: "shoot_rail"}.get(wpn, "shoot" if i == 0 or self.pvp else "shoot_ai")
            self.sound.play(name, x=self.sx(x), gain=1.0 if i == 0 or self.pvp else 0.8)
            if i == 0 and not self.pvp:
                self.cam.kick(a, z({1: 7, 3: 9}.get(wpn, 3)))
                self.cross_spread = min(self.cross_spread + {1: 10, 3: 12}.get(wpn, 5), 18)
        for s, t, x, y, absorbed in ev.hits:
            stats["hits"][min(s, 1)] += 1
            self.flash[t] = 7
            if absorbed:
                self.particles.burst(x, y, G.SHIELD_C, n=18, speed=5, life=22, spark=True)
                self.ftext.add("PAJZS", x, y - 24, G.SHIELD_C, 18)
                self.sound.play("shield", x=self.sx(x))
                self.cam.add_shake(2)
                continue
            self.particles.burst(x, y, colors[t], n=14, speed=6, life=20, spark=True)
            self.particles.burst(x, y, colors[t], n=8, speed=2.5, life=26, size=3)
            if t == 0 and not self.pvp:
                self.ftext.add("-1", x, y - 24, (255, 110, 120), 22)
                self.sound.play("hurt", x=self.sx(x))
                self.cam.add_shake(z(5))
                self.hurt_flash = 16
                self.freeze = max(self.freeze, 3)
                sx_, sy_ = w.pos[s] - w.pos[0]
                self.dmg_dirs.append([math.atan2(sy_, sx_), 45])
            else:
                self.ftext.add("TALÁLAT", x, y - 24, G.GOLD, 18)
                self.sound.play("hit", x=self.sx(x))
                self.cam.add_shake(z(2.5 if not self.pvp else 3))
                if s == 0 and not self.pvp:
                    self.freeze = max(self.freeze, 2)
        for i, x, y in ev.deaths:
            self.particles.burst(x, y, colors[i], n=40, speed=9, life=34, spark=True)
            self.particles.burst(x, y, colors[i], n=30, speed=4, life=55, size=5)
            self.rings.add(x, y, G.lighten(colors[i], 0.3), 110)
            self.rings.add(x, y, colors[i], 60)
            self.cam.add_shake(z(7))
            self.sound.play("death", x=self.sx(x))
        for s, x, y in ev.impacts:
            if self.cam.visible(x, y):
                self.particles.burst(x, y, colors[s], n=6, speed=3.5, life=12, spark=True)
                self.particles.burst(x, y, colors[s], n=1, speed=0.2, life=10, size=3)
        for a, kind, x, y in ev.pickups:
            self.sound.play("pickup", x=self.sx(x))
            self.ftext.add(G.ITEM_LABEL[kind] + ("!" if kind < 3 else ""), x, y - 30, G.ITEM_C[kind], 20)
            self.rings.add(x, y, G.ITEM_C[kind], 50)
            self.particles.burst(x, y, G.ITEM_C[kind], n=16, speed=4, life=24, spark=True)
        for dodger, shooter in ev.dodges:
            stats["dodges"][min(dodger, 1)] += 1
            if dodger == 0 and not self.pvp and self.frame - self.last_dodge_txt > 30:
                self.last_dodge_txt = self.frame
                self.sound.play("dodge", x=self.sx(w.pos[0, 0]))
                self.ftext.add("kitérés", *(w.pos[0] + (0, -28)), G.DIM, 15)

    def sx(self, wx):
        """World x -> screen x in the 0..800 range the stereo panner expects."""
        return self.cam.w2s(wx, 0)[0] / G.SW * 800

    def draw_arena(self, snap, colors, hp_pips=False):
        s = self.screen
        cam = self.cam
        G.draw_world(s, snap, colors, cam, self.flash, hp_pips=hp_pips, t=self.frame, trails=self.trails)
        for x, y, c, life in self.muzzles:
            sx, sy = cam.w2s(x, y)
            G.add_glow(s, G.lighten(c, 0.5), sx, sy, 26 * cam.zoom * life / 4, intensity=1.2)
        self.rings.draw(s, cam)
        self.particles.draw(s, cam)
        self.ftext.draw(s, cam)
        if self.hurt_flash:
            v = G.vignette().copy()
            v.set_alpha(int(255 * self.hurt_flash / 16))
            s.blit(v, (0, 0))
        if self.dmg_dirs and self.w.alive[0]:
            px, py = cam.w2s(*self.w.pos[0])
            for ang, life in self.dmg_dirs:
                k = life / 45
                rad = z(70)
                col = (int(255 * k), int(40 * k), int(60 * k))
                rect = pg.Rect(px - rad, py - rad, 2 * rad, 2 * rad)
                # pygame arcs go counter-clockwise with y up; screen angle -> arc angle is -ang
                pg.draw.arc(s, col, rect, -ang - 0.45, -ang + 0.45, z(5))
                G.add_glow(s, (255, 40, 60), px + math.cos(ang) * rad, py + math.sin(ang) * rad, z(30), intensity=0.7 * k)

    def low_hp_pulse(self, active):
        if active:
            v = G.vignette().copy()
            v.set_alpha(int(55 + 40 * math.sin(self.frame * 0.12)))
            self.screen.blit(v, (0, 0))

    def draw_crosshair(self, w):
        if self.bot:
            return
        s = self.screen
        x, y = pg.mouse.get_pos()
        sp = z(self.cross_spread)
        G.add_glow(s, G.CYAN, x, y, z(22), intensity=0.35)
        pg.draw.circle(s, G.CYAN, (x, y), z(9) + sp // 2, z(2))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pg.draw.line(s, G.lighten(G.CYAN, 0.5), (x + dx * (z(6) + sp), y + dy * (z(6) + sp)),
                         (x + dx * (z(15) + sp), y + dy * (z(15) + sp)), z(2))
        frac = 1 - w.cd[0] / C.W_COOLDOWN[int(w.weapon[0])]
        if frac < 1:
            r = z(16) + sp
            pg.draw.arc(s, G.GOLD, (x - r, y - r, 2 * r, 2 * r), math.pi / 2, math.pi / 2 + math.tau * max(frac, 0), z(2))

    def draw_debug(self, w, ai_moves):
        if not self.debug:
            return
        for idx, mv in ai_moves:
            if np.any(mv):
                x, y = self.cam.w2s(*w.pos[idx])
                ex, ey = x + mv[0] * z(42), y + mv[1] * z(42)
                pg.draw.line(self.screen, G.GOLD, (x, y), (ex, ey), z(2))
                pg.draw.circle(self.screen, G.GOLD, (int(ex), int(ey)), z(3))
        us = np.mean(self.infer_us) if self.infer_us else 0
        G.text(self.screen, f"AI GONDOLATAI  ·  sárga nyíl = a választott irány  ·  döntés {us:4.0f} µs/képkocka",
               self.f_mono, G.GOLD, topleft=P(10, 540 - 22))

    def pause_check(self, events):
        for e in events:
            if e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_p):
                self.paused_from = self.scene
                self.scene = "pause"
                self.focus = "resume"
                self.pressed = None
                pg.mouse.set_visible(True)
                self.sound.play("back")
                return True
            if e.type == pg.KEYDOWN and e.key == pg.K_TAB:
                self.debug = not self.debug
        return False


    def hud_bar(self):
        s = self.screen
        top = pg.Surface((G.SW, G.HUD_H), pg.SRCALPHA)
        top.fill((8, 9, 20, 200))
        s.blit(top, (0, 0))
        pg.draw.line(s, G.mix(G.CYAN, G.PINK, 0.35), (0, G.HUD_H - 1), (G.SW, G.HUD_H - 1), max(1, z(1)))


    def banner(self, title, sub, color, t):
        s = self.screen
        k = ease_out(t / 12)
        h = int(z(120) * k)
        cy = (G.SH + G.HUD_H) // 2
        band = pg.Surface((G.SW, max(h, 1)), pg.SRCALPHA)
        band.fill((6, 7, 16, 215))
        s.blit(band, (0, cy - h // 2))
        pg.draw.line(s, color, (0, cy - h // 2), (G.SW, cy - h // 2), z(1.5))
        pg.draw.line(s, color, (0, cy + h // 2), (G.SW, cy + h // 2), z(1.5))
        if k > 0.6:
            G.glow_text(s, title, self.f_big, G.lighten(color, 0.55), glow=color, spacing=4,
                        center=(G.SW // 2 + (1 - k) * z(200), cy - z(16)))
            if sub:
                G.text(s, sub, self.f_mid, G.TEXT, center=(G.SW // 2, cy + z(30)))

    def draw_countdown(self):
        s = self.screen
        n = 3 - self.timer // 30
        k = (self.timer % 30) / 30
        f = G.font(int(150 - 60 * ease_out(k)))
        img = f.render(str(n), True, G.GOLD)
        img.set_alpha(int(255 * (1 - k * 0.7)))
        r = img.get_rect(center=(G.SW // 2, (G.SH + G.HUD_H) // 2))
        G.add_glow(s, G.GOLD, r.centerx, r.centery, z(90), intensity=0.35 * (1 - k))
        s.blit(img, r)

    def draw_go(self):
        k = self.timer / 35
        G.glow_text(self.screen, "HARC!", self.f_title, G.GOLD, spacing=8, alpha=int(255 * (1 - k)),
                    center=(G.SW // 2, (G.SH + G.HUD_H) // 2 - z(20) * k))

    def draw_replay_overlay(self):
        s = self.screen
        y0 = G.HUD_H
        for y in (y0, G.SH - z(36)):
            pg.draw.rect(s, (0, 0, 0), (0, y, G.SW, z(36)))
        if (self.frame // 20) % 2:
            pg.draw.circle(s, (255, 50, 70), (z(26), y0 + z(18)), z(6))
        G.text(s, "VISSZAJÁTSZÁS", self.f_btn, G.TEXT, spacing=3, midleft=(z(42), y0 + z(18)))
        G.text(s, "½×", self.f_small, G.DIM, midleft=(z(232), y0 + z(18)))
        G.text(s, "Space: tovább", self.f_small, G.DIM, midright=(G.SW - z(16), G.SH - z(18)))

    def start_killcam(self, pos):
        self.killcam_pos = pos
        self.phase, self.timer = "killcam", 0

    def update_killcam(self):
        """Slow-motion beat on the killing blow before the replay."""
        self.update_fx(slow=True)
        if self.timer >= 50:
            self.frames = list(self.replay)
            self.replay_i = 0.0
            self.phase, self.timer = "replay", 0

    def replay_step(self, events):
        prev = int(self.replay_i)
        self.replay_i += REPLAY_SPEED
        cur = int(self.replay_i)
        skip = any((e.type == pg.KEYDOWN and e.key in (pg.K_SPACE, pg.K_RETURN)) or
                   (e.type == pg.MOUSEBUTTONUP and e.button == 1) for e in events)
        if cur >= len(self.frames) or skip:
            return True
        f = self.frames[cur]
        if cur != prev:
            a, b = self.frames[prev], f
            for k in np.flatnonzero(b.hp < a.hp):
                self.flash[k] = 7
                self.particles.burst(*b.pos[k], self.colors[k], n=14, speed=6, life=20, spark=True)
                self.sound.play("hurt" if k == 0 else "hit", x=self.sx(b.pos[k][0]), gain=0.6)
            for k in np.flatnonzero(a.alive & ~b.alive):
                self.particles.burst(*b.pos[k], self.colors[k], n=40, speed=9, life=34, spark=True)
                self.rings.add(*b.pos[k], self.colors[k], 110)
        return False

    def weapon_tag(self, snap, k, x, y, right=False):
        """'SÖRÉTES 6' + shield next to a player's name in the HUD."""
        parts = []
        wk = int(snap.weapon[k])
        if wk:
            parts.append((f"{G.WEAPON_LABEL[wk]} {int(snap.ammo[k])}", G.WEAPON_C[wk]))
        if snap.shield[k]:
            parts.append((f"PAJZS {int(snap.shield[k])}", G.SHIELD_C))
        for label, col in (reversed(parts) if right else parts):
            if right:
                r = G.text(self.screen, label, self.f_tiny, col, shadow=False, midright=(x, y))
                x = r.x - z(12)
            else:
                r = G.text(self.screen, label, self.f_tiny, col, shadow=False, midleft=(x, y))
                x = r.right + z(12)

    # ================================================================ duel / pvp
    def start_duel(self, pvp=False):
        self.pvp = pvp
        self.mode = "pvp" if pvp else "duel"
        self.diff = self.save["difficulty"]
        self.w = World([C.MAX_HP, C.MAX_HP], team=[0, 1], obstacles=self.current_map(), items=self.save["items"],
                       seed=self.args.seed)
        self.ai = None if pvp else AIController.from_difficulty(self.diff, seed=self.args.seed)
        self.bot2 = StraferAgent(seed=(self.args.seed or 0) + 1) if pvp and self.bot else None
        self.colors = [G.CYAN, G.PINK if pvp else G.ENEMY_C[self.diff]]
        self.score = [0, 0]
        self.stats = new_stats()
        self.round_no = 0
        self.ai_move = np.zeros(2)
        self.new_round()

    def new_round(self):
        gc.collect(1)
        if self.save["map"] == "random" and self.round_no > 0:
            self.w.set_map(self.current_map())
        self.w.reset_duel()
        self.facing = [float(self.w.aim[0]), float(self.w.aim[1])]
        self.replay = deque(maxlen=REPLAY_FRAMES)
        self.round_no += 1
        self.phase, self.timer = "countdown", 0
        self.clear_fx()
        self.cam = full_map_camera()
        self.cam.shake_on = self.save["shake"]
        for k in (0, 1):
            self.rings.add(*self.w.pos[k], self.colors[k], 60)

    def update_duel(self, events):
        if self.pause_check(events):
            return
        self.cam.update()
        self.timer += 1
        w = self.w
        if self.phase == "killcam":
            self.update_killcam()
            return
        self.update_fx()
        if self.phase == "countdown":
            if self.timer in (1, 31, 61):
                self.sound.play("count")
            if self.timer >= COUNTDOWN:
                self.sound.play("go")
                self.phase, self.timer = "fight", 0
        elif self.phase == "fight":
            if self.freeze > 0:          # hit-stop
                self.freeze -= 1
                return
            if self.pvp:
                pm, pa, ps = self.keyboard_player(w, 0)
                m2, a2, s2 = self.keyboard_player(w, 1)
            else:
                t0 = time.perf_counter()
                m, a, s, p = self.ai.act(duel_view(w, [1], t_cap=10 ** 9), ids=[1])
                self.infer_us.append((time.perf_counter() - t0) * 1e6)
                self.ai_move = m[0]
                m2, a2, s2 = m[0], a[0], s[0]
                pm, pa, ps = self.player_input(w)
            ev = w.step(np.stack([pm, m2]), np.array([pa, a2]), np.array([ps, s2]))
            self.handle_events(ev, self.colors, self.stats)
            self.update_trails(w)
            self.stats["frames"] += 1
            self.replay.append(w.snapshot())
            dead = ~w.alive
            if dead.any() or w.t >= C.MAX_TICKS:
                self.round_winner = -1 if dead.all() or not dead.any() else (0 if dead[1] else 1)
                if self.round_winner >= 0:
                    self.score[self.round_winner] += 1
                if dead.any():
                    self.start_killcam(w.pos[np.flatnonzero(dead)[0]].copy())
                else:
                    self.phase, self.timer = "roundover", 0
        elif self.phase == "replay":
            if self.replay_step(events):
                self.phase, self.timer = "roundover", 0
                self.sound.play("win" if self.pvp and self.round_winner >= 0 else
                                {0: "win", 1: "lose"}.get(self.round_winner, "count"))
        elif self.phase == "roundover":
            if self.timer >= 110:
                if max(self.score) >= WIN_ROUNDS:
                    self.finish_duel()
                else:
                    self.new_round()

    def finish_duel(self):
        won = self.score[0] > self.score[1]
        st = self.stats
        compare = [("TALÁLATI ARÁNY", acc(st, 0), acc(st, 1), True),
                   ("TALÁLATOK", st["hits"][0], st["hits"][1], False),
                   ("KITÉRÉSEK", st["dodges"][0], st["dodges"][1], False),
                   ("LÖVÉSEK", st["shots"][0], st["shots"][1], False)]
        if self.pvp:
            self.save["pvp"][0 if won else 1] += 1
            self.write_save()
            win_i = 0 if won else 1
            self.result = {
                "mode": "pvp", "title": f"{P_NAMES[win_i]} NYERT", "good": True, "color": self.colors[win_i],
                "labels": ("1. J.", "2. J."), "compare": compare,
                "sub": f"{self.score[0]} : {self.score[1]}   ·   {MAP_LABEL[self.save['map']]}",
                "foot": f"Összesítés: 1. játékos {self.save['pvp'][0]} – {self.save['pvp'][1]} 2. játékos",
            }
            self.sound.play("win")
        else:
            rec = self.save["duel"].setdefault(self.diff, [0, 0])
            rec[0 if won else 1] += 1
            self.write_save()
            self.result = {
                "mode": "duel", "title": "GYŐZELEM" if won else "VERESÉG", "good": won, "compare": compare,
                "sub": f"{self.score[0]} : {self.score[1]}   ·   Pip ({DIFFICULTIES[self.diff]['label']})   ·   "
                       f"{MAP_LABEL[self.save['map']]}",
                "foot": f"Mérleg ezen a szinten: {rec[0]} győzelem – {rec[1]} vereség",
            }
            self.sound.play("win" if won else "lose")
        self.goto("results")

    def draw_duel_hud(self, snap):
        s = self.screen
        self.hud_bar()
        names = P_NAMES if self.pvp else ["TE", f"PIP · {DIFFICULTIES[self.diff]['label'].upper()}"]
        y1, y2 = z(15), z(31)
        G.add_glow(s, G.CYAN, z(20), z(23), z(18), intensity=0.6)
        pg.draw.circle(s, G.CYAN, (z(20), z(23)), z(6))
        r = G.text(s, names[0], self.f_btn, G.TEXT, midleft=(z(34), y1))
        self.weapon_tag(snap, 0, r.right + z(14), y1 + z(1))
        G.seg_bar(s, z(34), y2 - z(5), z(220), z(10), snap.hp[0], C.MAX_HP, G.CYAN)
        col = self.colors[1]
        G.add_glow(s, col, G.SW - z(20), z(23), z(18), intensity=0.6)
        pg.draw.circle(s, col, (G.SW - z(20), z(23)), z(6))
        r = G.text(s, names[1], self.f_btn, G.TEXT, midright=(G.SW - z(34), y1))
        self.weapon_tag(snap, 1, r.x - z(14), y1 + z(1), right=True)
        G.seg_bar(s, G.SW - z(254), y2 - z(5), z(220), z(10), snap.hp[1], C.MAX_HP, col, right_align=True)
        cx = G.SW // 2
        for k, side in ((0, -1), (1, 1)):
            for i in range(WIN_ROUNDS):
                x = cx + side * z(34 + i * 20)
                on = i < self.score[k]
                if on:
                    G.add_glow(s, self.colors[k], x, z(15), z(14), intensity=0.7)
                pg.draw.circle(s, self.colors[k] if on else (40, 44, 74), (x, z(15)), z(6))
        secs = snap.t / C.FPS
        G.text(s, f"{self.round_no}. KÖR  ·  {int(secs) // 60}:{int(secs) % 60:02d}", self.f_tiny, G.DIM,
               shadow=False, center=(cx, z(35)))

    def draw_duel(self):
        replay = self.phase == "replay"
        snap = self.frames[min(int(self.replay_i), len(self.frames) - 1)] if replay else self.w
        self.draw_arena(snap, self.colors)
        if not replay:
            self.low_hp_pulse(not self.pvp and self.w.hp[0] == 1 and self.w.alive[0])
            if not self.pvp:
                self.draw_debug(self.w, [(1, self.ai_move)])
        self.draw_duel_hud(snap)
        if self.phase == "countdown":
            self.draw_countdown()
        elif self.phase == "fight" and self.timer < 35:
            self.draw_go()
        elif replay:
            self.draw_replay_overlay()
        elif self.phase == "roundover":
            if self.pvp:
                title, col = {0: (f"{P_NAMES[0]} VISZI A KÖRT", G.CYAN), 1: (f"{P_NAMES[1]} VISZI A KÖRT", G.PINK),
                              -1: ("DÖNTETLEN", G.DIM)}[self.round_winner]
            else:
                title, col = {0: ("KÖRT NYERTED", G.CYAN), 1: ("PIP NYERTE A KÖRT", self.colors[1]),
                              -1: ("DÖNTETLEN", G.DIM)}[self.round_winner]
            self.banner(title, f"{self.score[0]}  :  {self.score[1]}", col, self.timer)
        if self.phase in ("fight", "countdown") and not self.pvp:
            self.draw_crosshair(self.w)

    update_pvp = update_duel
    draw_pvp = draw_duel

    # ================================================================ survival
    def start_survival(self):
        self.pvp = False
        self.mode = "survival"
        self.w = World([SURV_PLAYER_HP] + [SURV_ENEMY_HP] * MAX_ENEMIES, obstacles=self.current_map(),
                       items=self.save["items"], seed=self.args.seed)
        self.w.pos[0] = self.w.ar.free_point(0, C.AGENT_R + 6, margin=400)
        self.w.alive[1:] = False
        self.w.hp[1:] = 0
        self.ctrls = {}
        self.enemy_diff = [None] * (MAX_ENEMIES + 1)
        self.colors = [G.CYAN] + [G.ENEMY_C["medium"]] * MAX_ENEMIES
        self.wave = 0
        self.points = 0
        self.shown_points = 0.0
        self.kills = 0
        self.stats = new_stats()
        self.replay = deque(maxlen=REPLAY_FRAMES)
        self.ai_moves = []
        self.clear_fx()
        self.cam = full_map_camera()
        self.cam.shake_on = self.save["shake"]
        self.next_wave()

    def ctrl(self, diff):
        if diff not in self.ctrls:
            self.ctrls[diff] = AIController.from_difficulty(diff, seed=self.args.seed)
        return self.ctrls[diff]

    def next_wave(self):
        self.wave += 1
        self.wave_hurt = 0
        for k, d in enumerate(wave_enemies(self.wave)):
            slot = k + 1
            self.enemy_diff[slot] = d
            self.colors[slot] = G.ENEMY_C[d]
            self.w.spawn(slot, min_dist=450)
            self.ctrl(d).forget([slot])
            self.rings.add(*self.w.pos[slot], G.ENEMY_C[d], 70)
        self.phase, self.timer = "wavestart", 0
        self.sound.play("wave")

    def update_survival(self, events):
        if self.pause_check(events):
            return
        self.cam.update()
        self.timer += 1
        self.shown_points += (self.points - self.shown_points) * 0.15
        w = self.w
        if self.phase == "killcam":
            self.update_killcam()
            return
        self.update_fx()
        if self.phase == "wavestart":
            if self.timer >= 100:
                self.phase, self.timer = "fight", 0
                self.sound.play("go")
        elif self.phase == "fight":
            if self.freeze > 0:
                self.freeze -= 1
                return
            N = w.N
            move, aim, shoot = np.zeros((N, 2)), w.aim.copy(), np.zeros(N, dtype=bool)
            self.ai_moves = []
            t0 = time.perf_counter()
            for d in set(self.enemy_diff[1:]) - {None}:
                ids = [i for i in range(1, N) if w.alive[i] and self.enemy_diff[i] == d]
                if ids:
                    m, a, s, _ = self.ctrl(d).act(duel_view(w, ids), ids=ids)
                    move[ids], aim[ids], shoot[ids] = m, a, s
                    self.ai_moves += list(zip(ids, m))
            self.infer_us.append((time.perf_counter() - t0) * 1e6)
            move[0], aim[0], shoot[0] = self.player_input(w)
            hp_before = w.hp[0]
            ev = w.step(move, aim, shoot)
            self.handle_events(ev, self.colors, self.stats)
            self.update_trails(w)
            self.wave_hurt += hp_before - w.hp[0]
            self.stats["frames"] += 1
            for i, x, y in ev.deaths:
                if i != 0:
                    pts = 100 * self.wave
                    self.points += pts
                    self.kills += 1
                    self.ftext.add(f"+{pts}", x, y - 34, G.GOLD, 24)
            self.replay.append(w.snapshot())
            if not w.alive[0]:
                self.start_killcam(w.pos[0].copy())
            elif not w.alive[1:].any():
                bonus = 250 * self.wave + (500 if self.wave_hurt == 0 else 0)
                self.points += bonus
                self.clear_bonus = bonus
                self.healed = int(min(SURV_HEAL, w.max_hp[0] - w.hp[0]))
                w.hp[0] += self.healed
                self.phase, self.timer = "waveclear", 0
                self.sound.play("win")
        elif self.phase == "waveclear":
            if self.timer >= 120:
                self.next_wave()
        elif self.phase == "replay":
            if self.replay_step(events):
                self.finish_survival()

    def finish_survival(self):
        record = self.points > self.save["best_survival"]
        if record:
            self.save["best_survival"] = int(self.points)
        self.save["best_wave"] = max(self.save["best_wave"], self.wave)
        self.write_save()
        st = self.stats
        secs = st["frames"] / C.FPS
        self.result = {
            "mode": "survival", "title": "ÚJ REKORD" if record else "ELESTÉL", "good": record,
            "sub": f"{self.points} pont   ·   {self.wave}. hullám   ·   {MAP_LABEL[self.save['map']]}",
            "rows": [("LEGYŐZÖTT ELLENSÉG", str(self.kills)),
                     ("TÚLÉLT IDŐ", f"{int(secs) // 60}:{int(secs) % 60:02d}"),
                     ("TALÁLATI ARÁNY", f"{acc(st, 0):.0%}"),
                     ("KITÉRÉSEK", str(st["dodges"][0])),
                     ("KAPOTT TALÁLAT", str(st["hits"][1]))],
            "foot": f"Rekord: {self.save['best_survival']} pont  ·  legjobb hullám: {self.save['best_wave']}",
        }
        self.sound.play("lose" if not record else "win")
        self.goto("results")

    def draw_survival(self):
        s = self.screen
        replay = self.phase == "replay"
        snap = self.frames[min(int(self.replay_i), len(self.frames) - 1)] if replay else self.w
        self.draw_arena(snap, self.colors, hp_pips=True)
        if not replay:
            self.low_hp_pulse(self.w.hp[0] <= 3 and self.w.alive[0])
            self.draw_debug(self.w, self.ai_moves)
        self.hud_bar()
        G.add_glow(s, G.CYAN, z(20), z(23), z(18), intensity=0.6)
        pg.draw.circle(s, G.CYAN, (z(20), z(23)), z(6))
        G.text(s, "TE", self.f_btn, G.TEXT, midleft=(z(34), z(15)))
        self.weapon_tag(snap, 0, z(64), z(16))
        G.seg_bar(s, z(34), z(26), z(250), z(10), snap.hp[0], SURV_PLAYER_HP, G.CYAN, gap=z(2))
        cx = G.SW // 2
        G.text(s, f"{self.wave}. HULLÁM", self.f_mid, G.TEXT, spacing=2, center=(cx, z(16)))
        left = int(snap.alive[1:].sum())
        G.text(s, f"{left} ellenség", self.f_tiny, G.DIM, shadow=False, center=(cx, z(36)))
        G.text(s, f"{int(round(self.shown_points))}", self.f_mid, G.GOLD, midright=(G.SW - z(16), z(16)))
        G.text(s, f"rekord {self.save['best_survival']}", self.f_tiny, G.DIM, shadow=False,
               midright=(G.SW - z(16), z(36)))
        if self.phase == "wavestart":
            counts = {}
            for d in wave_enemies(self.wave):
                counts[d] = counts.get(d, 0) + 1
            sub = "   ".join(f"{c}× {DIFFICULTIES[d]['label']}" for d, c in counts.items())
            self.banner(f"{self.wave}. HULLÁM", sub, G.TEXT, self.timer)
        elif self.phase == "waveclear":
            extra = "   ·   HIBÁTLAN +500" if self.wave_hurt == 0 else ""
            self.banner("HULLÁM TELJESÍTVE", f"+{self.clear_bonus} pont   ·   +{self.healed} élet{extra}", G.GOLD,
                        self.timer)
        elif self.phase == "fight" and self.timer < 35:
            self.draw_go()
        elif replay:
            self.draw_replay_overlay()
        if self.phase in ("fight", "wavestart", "waveclear"):
            self.draw_crosshair(self.w)

    # ================================================================ pause
    PAUSE = [("resume", "FOLYTATÁS"), ("restart", "ÚJRAKEZDÉS"), ("menu", "FŐMENÜ"), ("quit", "KILÉPÉS")]

    def pause_items(self):
        return [(k, R(480 - 160, 196 + i * 56, 320, 46)) for i, (k, _) in enumerate(self.PAUSE)]

    def update_pause(self, events):
        hit = self.nav(events, self.pause_items(), back="resume")
        if hit is None:
            return
        self.sound.play("select" if hit != "resume" else "back")
        if hit == "resume":
            self.scene = self.paused_from
            pg.mouse.set_visible(False)
        elif hit == "restart":
            self.goto(self.paused_from)
        elif hit == "menu":
            self.goto("menu")
        else:
            self.running = False

    def draw_pause(self):
        getattr(self, "draw_" + self.paused_from)()
        self.dim(200)
        s = self.screen
        G.glow_text(s, "SZÜNET", self.f_title, G.TEXT, glow=G.CYAN, spacing=8, center=P(480, 128))
        labels = dict(self.PAUSE)
        for k, r in self.pause_items():
            self.button(k, r, labels[k], color=G.PINK if k == "quit" else G.CYAN, align="center")
        G.text(s, "Esc: folytatás", self.f_tiny, G.FAINT, shadow=False, center=P(480, 440))

    # ================================================================ results
    def result_items(self):
        return [("again", R(480 - 230, 458, 220, 48)), ("menu", R(480 + 10, 458, 220, 48))]

    def update_results(self, events):
        self.ui_particles.update()
        hit = self.nav(events, self.result_items(), axis="x", back="menu")
        for e in events:
            if e.type == pg.KEYDOWN and e.key == pg.K_r:
                hit = "again"
        if hit == "again":
            self.sound.play("select")
            self.goto(self.result["mode"])
        elif hit == "menu":
            self.sound.play("back")
            self.goto("menu")
        if self.result["good"] and self.frame % 9 == 0:
            x, y = np.random.uniform([z(60), z(60)], [G.SW - z(60), z(200)])
            col = [G.GOLD, G.CYAN, G.PINK][(self.frame // 9) % 3]
            self.ui_particles.burst(x, y, col, n=22, speed=z(6), life=40, spark=True)

    def draw_results(self):
        s = self.screen
        G.draw_world(s, self.w, self.colors, full_map_camera(), t=self.frame)
        self.dim(215)
        self.ui_particles.draw(s, self.screen_cam)
        r = self.result
        cx = G.SW // 2
        col = r.get("color") or (G.GOLD if r["good"] else G.RED)
        k = ease_out(1 - self.fade_in / 255)
        G.glow_text(s, r["title"], self.f_hero, col, spacing=8, center=(cx, z(66 + 20 * (1 - k))))
        G.text(s, r["sub"], self.f_mid, G.TEXT, center=(cx, z(124)))
        box = R(480 - 300, 152, 600, 250)
        G.panel(s, box, border=G.FAINT, alpha=190)
        if r["mode"] in ("duel", "pvp"):
            la, lb = r.get("labels", ("TE", "PIP"))
            G.text(s, la, self.f_btn, G.CYAN, center=(box.x + z(70), box.y + z(26)))
            G.text(s, lb, self.f_btn, self.colors[1], center=(box.right - z(70), box.y + z(26)))
            for i, (label, a_, b_, pct) in enumerate(r["compare"]):
                y = box.y + z(66 + i * 44)
                G.text(s, label, self.f_tiny, G.DIM, shadow=False, center=(cx, y - z(14)))
                tot = max(a_ + b_, 1e-9)
                half = z(200)
                wa, wb = int(half * a_ / tot), int(half * b_ / tot)
                pg.draw.rect(s, (30, 34, 62), (cx - half, y, 2 * half, z(10)), border_radius=z(5))
                pg.draw.rect(s, G.CYAN, (cx - wa, y, wa, z(10)), border_top_left_radius=z(5), border_bottom_left_radius=z(5))
                pg.draw.rect(s, self.colors[1], (cx, y, wb, z(10)), border_top_right_radius=z(5),
                             border_bottom_right_radius=z(5))
                fa = f"{a_:.0%}" if pct else str(a_)
                fb = f"{b_:.0%}" if pct else str(b_)
                G.text(s, fa, self.f_btn, G.TEXT, midright=(cx - half - z(12), y + z(5)))
                G.text(s, fb, self.f_btn, G.TEXT, midleft=(cx + half + z(12), y + z(5)))
        else:
            for i, (label, v) in enumerate(r["rows"]):
                y = box.y + z(36 + i * 44)
                G.text(s, label, self.f_small, G.DIM, midleft=(box.x + z(40), y))
                G.text(s, v, self.f_mid, G.TEXT, midright=(box.right - z(40), y))
                if i < len(r["rows"]) - 1:
                    pg.draw.line(s, (30, 34, 62), (box.x + z(30), y + z(22)), (box.right - z(30), y + z(22)))
        G.text(s, r["foot"], self.f_small, G.DIM, center=(cx, z(428)))
        for kid, rr in self.result_items():
            self.button(kid, rr, "ÚJRA" if kid == "again" else "FŐMENÜ",
                        color=G.CYAN if kid == "again" else G.PINK, align="center")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["menu", "duel", "pvp", "survival"], default="menu")
    p.add_argument("--difficulty", choices=DIFF_KEYS, default=None)
    p.add_argument("--map", choices=MAP_KEYS, default=None)
    p.add_argument("--bot", choices=["strafer", "rule"], default=None, help="a scripted bot plays for you")
    p.add_argument("--frames", type=int, default=0, help="quit after N frames")
    p.add_argument("--fast", action="store_true", help="don't cap at 60 fps (testing)")
    p.add_argument("--screenshot", default="")
    p.add_argument("--shots-dir", default="")
    p.add_argument("--shots-every", type=int, default=0)
    p.add_argument("--mute", action="store_true")
    p.add_argument("--windowed", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    Game(args).run()
    sys.exit(0)


if __name__ == "__main__":
    main()
