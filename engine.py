"""Headless, batched arena simulation.

State is a set of numpy arrays with a leading batch (env) dimension, so the same code runs one
interactive game (n_envs=1) or hundreds of training games at once. Agent axis: A agents
(training uses A=2; survival uses 1 player + up to 6 enemies). Projectile arrays are indexed by
*shooter*. Every env has its own obstacle map; unused obstacle slots are parked outside the arena.

Features: per-weapon cooldown/speed/pellets/damage/range, ammo, shields that absorb hits, and
pickups (weapons, shield, heal) that spawn at free spots.
"""
from dataclasses import dataclass

import numpy as np

import config as C

FAR_RECT = np.array([-1e4, -1e4, -1e4 + 1, -1e4 + 1])
W_COOLDOWN = np.array(C.W_COOLDOWN)
W_SPEED = np.array(C.W_SPEED)
W_PELLETS = np.array(C.W_PELLETS)
W_SPREAD = np.radians(C.W_SPREAD_DEG)
W_DAMAGE = np.array(C.W_DAMAGE)
W_LIFE = np.array(C.W_LIFE)
W_AMMO = np.array(C.W_AMMO)
ITEM_P = np.array(C.ITEM_WEIGHTS) / np.sum(C.ITEM_WEIGHTS)


@dataclass
class StepResult:
    hits: np.ndarray      # (n, A) damage dealt by each agent this tick (incl. shield-absorbed)
    taken: np.ndarray     # (n, A) damage received
    proj_hit: np.ndarray  # (n, A, P) which projectile slots hit this tick
    hit_target: np.ndarray  # (n, A, P) index of the agent hit, -1 if none
    fired: np.ndarray     # (n, A, P) which slots received a new projectile this tick
    picked: np.ndarray    # (n, A) item type + 1 picked up this tick, 0 = none
    died: np.ndarray      # (n, A) agents that died this tick
    ended: np.ndarray     # (n, A, P) shots that disappeared this tick without hitting
    ended_pmin: np.ndarray  # (n, A, P) closest approach of those shots to an enemy
    done: np.ndarray      # (n,) round over (one team left or timeout)
    winner: np.ndarray    # (n,) winning team, or -1 for draw / not done


def circle_rect_dist(p, rect):
    """Distance from points p (..., 2) to axis-aligned rect(s) (..., 4), 0 if inside."""
    closest = np.clip(p, rect[..., :2], rect[..., 2:])
    return np.linalg.norm(p - closest, axis=-1)


def push_out_of_rect(p, rect, r):
    """Move circles of radius r centred at p (..., 2) so they don't overlap rect (broadcastable (..., 4))."""
    lo, hi = rect[..., :2], rect[..., 2:]
    closest = np.clip(p, lo, hi)
    d = p - closest
    dist = np.linalg.norm(d, axis=-1)
    touching = (dist < r) & (dist > 0)
    p = np.where(touching[..., None], closest + d / np.maximum(dist, 1e-9)[..., None] * r, p)
    inside = dist == 0
    if inside.any():
        # centre inside the rect: push out along the axis of least penetration
        pen = np.stack(np.broadcast_arrays(p[..., 0] - lo[..., 0], hi[..., 0] - p[..., 0],
                                           p[..., 1] - lo[..., 1], hi[..., 1] - p[..., 1]), -1)
        k = pen.argmin(-1)
        q = p.copy()
        q[..., 0] = np.where(k == 0, lo[..., 0] - r, np.where(k == 1, hi[..., 0] + r, q[..., 0]))
        q[..., 1] = np.where(k == 2, lo[..., 1] - r, np.where(k == 3, hi[..., 1] + r, q[..., 1]))
        p = np.where(inside[..., None], q, p)
    return p


class Arena:
    def __init__(self, n_envs=1, n_agents=2, max_hp=C.MAX_HP, team=None, obstacles=None, items=False,
                 map_fn=None, seed=None):
        """obstacles: rect list shared by all envs (default config.OBSTACLES).
        map_fn: optional rng -> rect list, called per env on every reset (training map variety).
        team: team id per agent (default: every agent on its own team)."""
        self.n = n = n_envs
        self.A = A = n_agents
        P, K, I = C.MAX_PROJ, C.MAX_OBST, C.MAX_ITEMS
        self.rng = np.random.default_rng(seed)
        self.max_hp = np.broadcast_to(np.asarray(max_hp, dtype=np.int32), (A,)).copy()
        self.team = np.arange(A) if team is None else np.asarray(team)
        self.items_on = items
        self.map_fn = map_fn
        self.spawn_dist = C.SPAWN_MIN_DIST if A == 2 else 120
        self.pos = np.zeros((n, A, 2))
        self.vel = np.zeros((n, A, 2))
        self.aim = np.zeros((n, A))
        self.hp = np.zeros((n, A), dtype=np.int32)
        self.cd = np.zeros((n, A), dtype=np.int32)
        self.alive = np.ones((n, A), dtype=bool)
        self.weapon = np.zeros((n, A), dtype=np.int32)
        self.ammo = np.zeros((n, A), dtype=np.int32)
        self.shield = np.zeros((n, A), dtype=np.int32)
        self.ppos = np.zeros((n, A, P, 2))
        self.pvel = np.zeros((n, A, P, 2))
        self.pact = np.zeros((n, A, P), dtype=bool)
        self.pdmg = np.ones((n, A, P), dtype=np.int32)
        self.plife = np.zeros((n, A, P), dtype=np.int32)
        self.pmin = np.full((n, A, P), np.inf)   # closest approach of each shot to an enemy (dodge stats)
        self.ipos = np.zeros((n, I, 2))
        self.itype = np.zeros((n, I), dtype=np.int32)
        self.iact = np.zeros((n, I), dtype=bool)
        self.item_timer = np.zeros(n, dtype=np.int32)
        self.t = np.zeros(n, dtype=np.int32)
        self.obstacles = np.tile(FAR_RECT, (n, K, 1))
        self.k_used = 0
        rects = C.OBSTACLES if obstacles is None else obstacles
        if len(rects):
            self.set_map(np.arange(n), rects)
        self.reset()

    # ------------------------------------------------------------------ maps
    def set_map(self, idx, rects):
        rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        assert len(rects) <= C.MAX_OBST
        o = np.tile(FAR_RECT, (C.MAX_OBST, 1))
        o[:len(rects)] = rects
        self.obstacles[np.atleast_1d(idx)] = o
        # real rects always fill the leading slots, so loops only need the first k_used slots
        self.k_used = int((self.obstacles[:, :, 0] > -1e3).sum(1).max())

    def map_rects(self, e=0):
        o = self.obstacles[e]
        return o[o[:, 0] > -1e3]

    def _clear_of_walls(self, e, p, clearance):
        return (circle_rect_dist(np.asarray(p), self.obstacles[e]) > clearance).all()

    def free_point(self, e, clearance, avoid=(), min_dist=0.0, margin=None):
        """Random point in env e that is `clearance` away from walls and `min_dist` from `avoid`."""
        m = (C.AGENT_R + C.SPAWN_MARGIN) if margin is None else margin
        p = None
        for _ in range(300):
            p = self.rng.uniform([m, m], [C.W - m, C.H - m])
            if self._clear_of_walls(e, p, clearance) and all(np.linalg.norm(p - a) >= min_dist for a in avoid):
                return p
        return p

    # ------------------------------------------------------------------ reset
    def reset(self, mask=None):
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        if idx.size == 0:
            return
        if self.map_fn is not None:
            for e in idx:
                self.set_map(e, self.map_fn(self.rng))
        self.pos[idx] = self._spawn(idx)
        self.vel[idx] = 0
        self.hp[idx] = self.max_hp
        self.cd[idx] = 0
        self.alive[idx] = True
        self.weapon[idx] = 0
        self.ammo[idx] = 0
        self.shield[idx] = 0
        self.pact[idx] = False
        self.iact[idx] = False
        self.item_timer[idx] = C.ITEM_FIRST
        self.t[idx] = 0
        for a in range(self.A):
            b = 1 if a == 0 else 0
            d = self.pos[idx, b] - self.pos[idx, a]
            self.aim[idx, a] = np.arctan2(d[:, 1], d[:, 0])

    def _spawn(self, idx):
        k, A = idx.size, self.A
        out = np.empty((k, A, 2))
        todo = np.arange(k)
        m = C.AGENT_R + C.SPAWN_MARGIN
        while todo.size:
            p = self.rng.uniform([m, m], [C.W - m, C.H - m], size=(todo.size, A, 2))
            ok = np.ones(todo.size, dtype=bool)
            for a in range(A):
                for b in range(a + 1, A):
                    dab = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
                    ok &= dab >= self.spawn_dist
                    if A == 2:
                        ok &= dab <= C.SPAWN_MAX_DIST
            obs = self.obstacles[idx[todo]]
            for kk in range(self.k_used):
                ok &= (circle_rect_dist(p, obs[:, kk][:, None, :]) > C.AGENT_R + 5).all(axis=1)
            out[todo[ok]] = p[ok]
            todo = todo[~ok]
        return out

    def spawn_agent(self, e, i, min_dist=260):
        """(Re)spawn one agent (survival waves) away from the other living agents."""
        others = [self.pos[e, j] for j in range(self.A) if j != i and self.alive[e, j]]
        self.pos[e, i] = self.free_point(e, C.AGENT_R + 6, others, min_dist)
        self.vel[e, i] = 0
        self.hp[e, i] = self.max_hp[i]
        self.cd[e, i] = 0
        self.alive[e, i] = True
        self.weapon[e, i] = self.ammo[e, i] = self.shield[e, i] = 0
        self.pact[e, i] = False
        d = self.pos[e, 0] - self.pos[e, i]
        self.aim[e, i] = np.arctan2(d[1], d[0])

    # ------------------------------------------------------------------- step
    def _resolve_agents(self, p):
        r = C.AGENT_R
        for _ in range(2):
            p[..., 0] = np.clip(p[..., 0], r, C.W - r)
            p[..., 1] = np.clip(p[..., 1], r, C.H - r)
            for k in range(self.k_used):
                p = push_out_of_rect(p, self.obstacles[:, k][:, None, :], r)
            for a in range(self.A):
                for b in range(a + 1, self.A):
                    both = (self.alive[:, a] & self.alive[:, b])[:, None]
                    d = p[:, b] - p[:, a]
                    dist = np.linalg.norm(d, axis=-1, keepdims=True)
                    overlap = np.maximum(2 * r - dist, 0) * both
                    push = d / np.maximum(dist, 1e-9) * overlap / 2
                    p[:, a] -= push
                    p[:, b] += push
        return p

    def _items_step(self, picked):
        self.item_timer -= 1
        due = self.item_timer <= 0
        full = self.iact.sum(1) >= C.MAX_ITEMS
        for e in np.flatnonzero(due & ~full):
            slot = int(np.argmin(self.iact[e]))
            avoid = [self.pos[e, a] for a in range(self.A) if self.alive[e, a]]
            self.ipos[e, slot] = self.free_point(e, C.ITEM_R + C.AGENT_R + 4, avoid, 90, margin=50)
            self.itype[e, slot] = self.rng.choice(len(ITEM_P), p=ITEM_P)
            self.iact[e, slot] = True
        self.item_timer[due] = C.ITEM_EVERY
        d = np.linalg.norm(self.pos[:, :, None, :] - self.ipos[:, None, :, :], axis=-1)   # (n, A, I)
        can = self.alive[:, :, None] & self.iact[:, None, :] & (d < C.AGENT_R + C.ITEM_R)
        if not can.any():
            return
        for e, i in zip(*np.nonzero(can.any(1))):
            a = int(np.argmin(np.where(can[e, :, i], d[e, :, i], np.inf)))
            it = int(self.itype[e, i])
            if it <= 2:                       # weapons
                self.weapon[e, a] = it + 1
                self.ammo[e, a] = W_AMMO[it + 1]
            elif it == 3:
                self.shield[e, a] = min(self.shield[e, a] + C.SHIELD_ADD, C.SHIELD_MAX)
            else:
                self.hp[e, a] = min(self.hp[e, a] + C.HEAL_ADD, self.max_hp[a])
            self.iact[e, i] = False
            picked[e, a] = it + 1

    def step(self, move, aim, shoot):
        """move: (n, A, 2) direction vectors (clipped to length 1), aim: (n, A) absolute angles
        in radians, shoot: (n, A) bools. Dead agents' inputs are ignored."""
        n, A = self.n, self.A
        move = np.asarray(move, dtype=np.float64).reshape(n, A, 2)
        norm = np.linalg.norm(move, axis=-1, keepdims=True)
        move = np.where(norm > 1, move / np.maximum(norm, 1e-9), move)
        move = move * self.alive[..., None]
        self.aim = np.where(self.alive, np.asarray(aim, dtype=np.float64).reshape(n, A), self.aim)
        shoot = np.asarray(shoot, dtype=bool).reshape(n, A) & self.alive

        # agents
        self.vel += (move * C.AGENT_SPEED - self.vel) * C.AGENT_ACCEL
        old = self.pos
        new = self._resolve_agents(old + self.vel)
        self.vel = (new - old) * self.alive[..., None]
        self.pos = new

        # projectiles: swept segment vs every living agent of another team
        start = self.ppos
        end = start + self.pvel
        seg = end - start
        seg2 = np.maximum((seg * seg).sum(-1), 1e-9)
        hit_by = np.full(self.pact.shape, -1)
        enemy = self.team[:, None] != self.team[None, :]           # (shooter, target)
        for tgt in range(A):
            cand = self.pact & enemy[:, tgt][None, :, None] & self.alive[:, tgt][:, None, None] & (hit_by < 0)
            if not cand.any():
                continue
            tp = self.pos[:, tgt][:, None, None, :]
            u = np.clip(((tp - start) * seg).sum(-1) / seg2, 0, 1)
            dist = np.linalg.norm(tp - (start + seg * u[..., None]), axis=-1)
            self.pmin = np.where(cand, np.minimum(self.pmin, dist), self.pmin)
            hit_by[cand & (dist <= C.AGENT_R + C.PROJ_R)] = tgt
        proj_hit = hit_by >= 0
        self.ppos = end
        self.plife -= 1
        x, y = end[..., 0], end[..., 1]
        alive_p = self.pact & ~proj_hit & (self.plife > 0)
        alive_p &= (x >= -C.PROJ_R) & (x <= C.W + C.PROJ_R) & (y >= -C.PROJ_R) & (y <= C.H + C.PROJ_R)
        r = C.PROJ_R
        for k in range(self.k_used):
            o = self.obstacles[:, k][:, None, None, :]
            alive_p &= ~((x > o[..., 0] - r) & (x < o[..., 2] + r) & (y > o[..., 1] - r) & (y < o[..., 3] + r))
        ended = self.pact & ~alive_p & ~proj_hit
        ended_pmin = np.where(ended, self.pmin, np.inf)
        self.pact = alive_p

        # damage (shields absorb first)
        dmg = np.where(proj_hit, self.pdmg, 0)
        hits = dmg.sum(-1).astype(np.int32)
        taken = np.zeros((n, A), dtype=np.int32)
        for tgt in range(A):
            taken[:, tgt] = (dmg * (hit_by == tgt)).sum((1, 2))
        absorbed = np.minimum(self.shield, taken)
        self.shield -= absorbed
        self.hp -= taken - absorbed
        died = self.alive & (self.hp <= 0)
        self.alive &= ~died

        # pickups
        picked = np.zeros((n, A), dtype=np.int32)
        if self.items_on:
            self._items_step(picked)

        # firing
        self.cd = np.maximum(self.cd - 1, 0)
        fire = shoot & (self.cd == 0) & self.alive
        fired = np.zeros_like(self.pact)
        if fire.any():
            e, a = np.nonzero(fire)
            w = self.weapon[e, a]
            npel = W_PELLETS[w]
            for j in range(int(npel.max())):
                sel = npel > j
                ee, aa, ww = e[sel], a[sel], w[sel]
                slot = np.argmin(self.pact[ee, aa], axis=-1)       # first free slot
                ok = ~self.pact[ee, aa, slot]
                ee, aa, ww, slot = ee[ok], aa[ok], ww[ok], slot[ok]
                ang = self.aim[ee, aa] + (j - (W_PELLETS[ww] - 1) / 2) * W_SPREAD[ww]
                d = np.stack([np.cos(ang), np.sin(ang)], -1)
                self.ppos[ee, aa, slot] = self.pos[ee, aa] + d * (C.AGENT_R + C.PROJ_R + 1)
                self.pvel[ee, aa, slot] = d * W_SPEED[ww][:, None]
                self.pact[ee, aa, slot] = True
                self.pdmg[ee, aa, slot] = W_DAMAGE[ww]
                self.plife[ee, aa, slot] = W_LIFE[ww]
                self.pmin[ee, aa, slot] = np.inf
                fired[ee, aa, slot] = True
            self.cd[e, a] = W_COOLDOWN[w]
            special = w > 0
            if special.any():
                es, as_ = e[special], a[special]
                self.ammo[es, as_] -= 1
                empty = self.ammo[es, as_] <= 0
                self.weapon[es[empty], as_[empty]] = 0

        # outcome: a team wins when it is the only one with living agents
        self.t += 1
        teams = np.unique(self.team)
        team_alive = np.stack([self.alive[:, self.team == tm].any(1) for tm in teams], 1)
        n_alive = team_alive.sum(1)
        done = (n_alive <= 1) | (self.t >= C.MAX_TICKS)
        winner = np.where(n_alive == 1, teams[np.argmax(team_alive, 1)], -1)
        return StepResult(hits=hits, taken=taken, proj_hit=proj_hit, hit_target=hit_by, fired=fired,
                          picked=picked, died=died, ended=ended, ended_pmin=ended_pmin, done=done, winner=winner)
