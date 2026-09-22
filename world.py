"""Play-time world: one engine.Arena env with N agents (player = index 0), exposed without the
batch axis, plus per-step events for effects/sound, 1v1 views for the AI, and the AI controller
with difficulty handicaps.

The game runs on exactly the same engine code as training. Each enemy sees the world as a 1v1
duel against its target via `pair_view`, which is what makes multi-enemy survival work without
retraining.
"""
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import config as C
from encoding import AIM_OFFSETS, MOVE_DIRS, encode
from engine import Arena
from model import FastPolicy, get_device, load_actor

NEAR_MISS = 45.0  # a shot that passes this close without hitting counts as dodged


@dataclass
class Events:
    fired: list = field(default_factory=list)    # (agent, x, y, angle, weapon)
    hits: list = field(default_factory=list)     # (shooter, target, x, y, absorbed_by_shield)
    deaths: list = field(default_factory=list)   # (agent, x, y)
    dodges: list = field(default_factory=list)   # (dodger, shooter)
    pickups: list = field(default_factory=list)  # (agent, item_type, x, y)
    impacts: list = field(default_factory=list)  # (shooter, x, y) shots that hit a wall / the arena edge


def _view(name):
    return property(lambda self: getattr(self.ar, name)[0])


class World:
    pos, vel, aim, hp, cd, alive = (_view(k) for k in ("pos", "vel", "aim", "hp", "cd", "alive"))
    weapon, ammo, shield = (_view(k) for k in ("weapon", "ammo", "shield"))
    ppos, pvel, pact, pdmg = (_view(k) for k in ("ppos", "pvel", "pact", "pdmg"))
    ipos, itype, iact = (_view(k) for k in ("ipos", "itype", "iact"))

    def __init__(self, max_hp, team=None, obstacles=(), items=True, seed=None):
        """max_hp: per-agent HP list (index 0 = player). team: default player vs everyone else."""
        N = len(max_hp)
        team = [0] + [1] * (N - 1) if team is None else team
        self.ar = Arena(1, n_agents=N, max_hp=max_hp, team=team, obstacles=list(obstacles), items=items, seed=seed)
        self.N = N
        self.max_hp = self.ar.max_hp
        self.team = self.ar.team

    @property
    def t(self):
        return int(self.ar.t[0])

    @t.setter
    def t(self, v):
        self.ar.t[0] = v

    @property
    def obstacles(self):
        return self.ar.map_rects(0)

    def set_map(self, rects):
        self.ar.set_map(0, rects)

    def reset_duel(self):
        self.ar.reset()

    def spawn(self, i, min_dist=260):
        self.ar.spawn_agent(0, i, min_dist)

    def step(self, move, aim, shoot):
        ar = self.ar
        shield_before = ar.shield[0].copy()
        res = ar.step(np.asarray(move, dtype=np.float64)[None], np.asarray(aim, dtype=np.float64)[None],
                      np.asarray(shoot, dtype=bool)[None])
        ev = Events()
        for a, k in zip(*np.nonzero(res.fired[0])):
            v = ar.pvel[0, a, k]
            ev.fired.append((a, *ar.ppos[0, a, k], math.atan2(v[1], v[0]), int(ar.weapon[0, a])))
        absorbed = shield_before - ar.shield[0]
        for s, k in zip(*np.nonzero(res.proj_hit[0])):
            tgt = int(res.hit_target[0, s, k])
            ev.hits.append((s, tgt, *ar.pos[0, tgt], bool(absorbed[tgt] > 0)))
        for i in np.flatnonzero(res.died[0]):
            ev.deaths.append((i, *ar.pos[0, i]))
        for s, k in zip(*np.nonzero(res.ended[0] & (res.ended_pmin[0] < NEAR_MISS))):
            p = ar.ppos[0, s, k]
            cand = np.flatnonzero(self.team != self.team[s])
            ev.dodges.append((int(cand[np.argmin(np.linalg.norm(ar.pos[0, cand] - p, axis=-1))]), s))
        for s, k in zip(*np.nonzero(res.ended[0] & (ar.plife[0] > 0) & ~res.fired[0])):
            ev.impacts.append((s, *ar.ppos[0, s, k]))
        for a in np.flatnonzero(res.picked[0]):
            ev.pickups.append((a, int(res.picked[0, a]) - 1, *ar.pos[0, a]))
        return ev

    def snapshot(self):
        """Copy of everything the renderer needs (used for replays)."""
        ar = self.ar
        return SimpleNamespace(pos=ar.pos[0].copy(), aim=ar.aim[0].copy(), hp=ar.hp[0].copy(),
                               max_hp=self.max_hp.copy(), cd=ar.cd[0].copy(), alive=ar.alive[0].copy(),
                               team=self.team.copy(), ppos=ar.ppos[0].copy(), pvel=ar.pvel[0].copy(),
                               pact=ar.pact[0].copy(), pdmg=ar.pdmg[0].copy(), weapon=ar.weapon[0].copy(),
                               ammo=ar.ammo[0].copy(), shield=ar.shield[0].copy(), ipos=ar.ipos[0].copy(),
                               itype=ar.itype[0].copy(), iact=ar.iact[0].copy(), t=self.t,
                               obstacles=self.obstacles)


def pair_view(w, mes, them, t_cap=C.MAX_TICKS // 2):
    """Batch of 1v1 views: row r has slot 0 = agent mes[r], slot 1 = agent `them`, and slot 1's
    projectiles are the incoming ones. HP is rescaled to the trained 0..MAX_HP range. The
    result quacks like a 2-agent engine.Arena for encode/decode_action and the scripted bots."""
    ar = w.ar
    e = np.atleast_1d(np.asarray(mes))
    n = e.size
    P = ar.pact.shape[2]
    v = SimpleNamespace(n=n, obstacles=ar.obstacles[0, :max(ar.k_used, 0)], k_used=ar.k_used)

    def pair(x):
        x = x[0]
        return np.stack([x[e], np.broadcast_to(x[them], x[e].shape)], 1)
    v.pos, v.vel, v.cd = pair(ar.pos), pair(ar.vel), pair(ar.cd)
    v.weapon, v.ammo, v.shield = pair(ar.weapon), pair(ar.ammo), pair(ar.shield)
    hp = np.ceil(ar.hp[0] / ar.max_hp * C.MAX_HP).astype(np.int32)[None]
    v.hp = pair(hp)
    v.ppos = np.zeros((n, 2, P, 2))
    v.pvel = np.zeros((n, 2, P, 2))
    v.pact = np.zeros((n, 2, P), dtype=bool)
    v.ppos[:, 1], v.pvel[:, 1], v.pact[:, 1] = ar.ppos[0, them], ar.pvel[0, them], ar.pact[0, them]
    v.ipos = np.broadcast_to(ar.ipos[0], (n,) + ar.ipos.shape[1:])
    v.itype = np.broadcast_to(ar.itype[0], (n,) + ar.itype.shape[1:])
    v.iact = np.broadcast_to(ar.iact[0], (n,) + ar.iact.shape[1:])
    v.t = np.full(n, min(int(ar.t[0]), t_cap), dtype=np.int32)
    return v


def duel_view(w, enemies, t_cap=C.MAX_TICKS // 2):
    """One view per enemy, each facing the player (index 0)."""
    return pair_view(w, enemies, 0, t_cap)


# ---------------------------------------------------------------- AI + difficulty
DIFFICULTIES = {
    # key: label, checkpoint, sample vs argmax, reaction delay (frames), aim noise (deg).
    # v2 = trained on the 1920x1080 maps with cover and pickups (warm-started from v1).
    "beginner":   dict(label="Kezdő",      ckpt="runs/v2/ckpt_0025.pt", stochastic=True,  delay=15, noise=10),
    "easy":       dict(label="Könnyű",     ckpt="runs/v2/ckpt_0100.pt", stochastic=True,  delay=8,  noise=5),
    "medium":     dict(label="Közepes",    ckpt="runs/v2/best.pt",      stochastic=True,  delay=4,  noise=0),
    "hard":       dict(label="Nehéz",      ckpt="runs/v2/best.pt",      stochastic=True,  delay=2,  noise=0),
    "impossible": dict(label="Lehetetlen", ckpt="runs/v2/best.pt",      stochastic=False, delay=0,  noise=0),
}
_actor_cache = {}


class AIController:
    """Trained policy + human-like handicaps: it acts on what it saw `delay` frames ago and
    adds Gaussian aim noise. Rows are identified by ids so enemies can come and go."""

    def __init__(self, ckpt, stochastic=False, delay=0, noise=0.0, seed=None):
        if ckpt not in _actor_cache:
            _actor_cache[ckpt] = load_actor(ckpt, get_device("cpu"))[0]
        self.policy = FastPolicy(_actor_cache[ckpt], deterministic=not stochastic, seed=seed)
        self.delay, self.noise = delay, math.radians(noise)
        self.rng = np.random.default_rng(seed)
        self.ring_obs = self.ring_base = None
        self.fresh = np.zeros(0, dtype=bool)
        self.ptr = 0

    @classmethod
    def from_difficulty(cls, key, seed=None):
        d = DIFFICULTIES[key]
        return cls(d["ckpt"], d["stochastic"], d["delay"], d["noise"], seed)

    def forget(self, ids):
        """Call when the agents behind these ids (re)spawn."""
        if self.fresh.size:
            ids = np.asarray(ids)
            self.fresh[ids[ids < self.fresh.size]] = True

    def act(self, view, i=0, ids=None):
        """-> move (n,2), absolute aim (n,), shoot (n,), shoot probability (n,)."""
        ids = np.arange(view.n) if ids is None else np.asarray(ids)
        obs = encode(view, i)
        d = view.pos[:, 1 - i] - view.pos[:, i]
        base = np.arctan2(d[:, 1], d[:, 0])
        if self.delay:
            K = self.delay + 1
            if self.ring_obs is None or self.ring_obs.shape[1] <= ids.max():
                size = max(int(ids.max()) + 1, 8)
                ro, rb = np.zeros((K, size, obs.shape[1]), np.float32), np.zeros((K, size))
                fr = np.ones(size, dtype=bool)
                if self.ring_obs is not None:
                    old = self.ring_obs.shape[1]
                    ro[:, :old], rb[:, :old], fr[:old] = self.ring_obs, self.ring_base, self.fresh
                self.ring_obs, self.ring_base, self.fresh = ro, rb, fr
            new = self.fresh[ids] | (np.asarray(view.t) == 0)
            if new.any():
                self.ring_obs[:, ids[new]] = obs[new]
                self.ring_base[:, ids[new]] = base[new]
                self.fresh[ids[new]] = False
            self.ring_obs[self.ptr, ids] = obs
            self.ring_base[self.ptr, ids] = base
            self.ptr = (self.ptr + 1) % K
            obs, base = self.ring_obs[self.ptr, ids], self.ring_base[self.ptr, ids]  # oldest
        m, a, s, p = self.policy.decide(obs)
        aim = base + AIM_OFFSETS[a]
        if self.noise:
            aim = aim + self.rng.normal(0, self.noise, aim.shape)
        return MOVE_DIRS[m], aim, s.astype(bool), p
