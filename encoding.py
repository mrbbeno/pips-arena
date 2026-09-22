"""94-float egocentric state encoding + discrete action decoding.

The first 67 features are exactly the v1 encoding (so a v1 network can be warm-started by
zero-padding its first layer), followed by 27 v2 features for weapons, shields and pickups:
  [0:6]   self: x, y (normalised to [-1,1]), vx, vy (/speed), hp frac, cooldown frac
  [6:15]  opponent: dx/800, dy/600, dist/1000, cos, sin (bearing), vx, vy, hp frac, cooldown frac
  [15]    line of sight (1 = clear)
  [16:24] 8 ray distances to wall/obstacle (/1000, capped at 1), directions 0,45,...,315 deg
  [24:66] 6 incoming enemy projectiles x (dx/800, dy/600, vx, vy, miss dist, time-to-closest, present)
  [66]    fraction of round time remaining
  [67:72] self: weapon one-hot (shotgun, rapid, rail), ammo fraction, shield fraction
  [72:76] opponent: weapon one-hot (shotgun, rapid, rail), shield fraction
  [76:94] 2 nearest pickups x (dx/800, dy/600, dist/1000, type one-hot x5, present)
Absolute position (features 0-1) is normalised by the actual map size.

Works on any object with the Arena attributes and exactly two agents (an engine.Arena with
A=2 or a world.pair_view).
"""
import numpy as np

import config as C

OBS_DIM = 94
OBS_DIM_V1 = 67
N_MOVE, N_AIM = 9, 7
K_PROJ = 6
PROJ_FEATS = 7
K_ITEMS = 2
ITEM_FEATS = 9

DIAG = C.REF_DIAG            # relative features use a fixed length scale, so a fight looks
RW, RH = C.REF_W, C.REF_H     # the same to the network on any map size (v1 was trained at 800x600)
_s = np.sqrt(0.5)
MOVE_DIRS = np.array([[0, 0], [1, 0], [_s, _s], [0, 1], [-_s, _s],
                      [-1, 0], [-_s, -_s], [0, -1], [_s, -_s]])
AIM_OFFSETS = np.radians([-30, -20, -10, 0, 10, 20, 30])
_ray_ang = np.arange(8) * np.pi / 4
RAY_DIRS = np.stack([np.cos(_ray_ang), np.sin(_ray_ang)], -1)
_W_AMMO = np.maximum(np.array(C.W_AMMO), 1)

MISS_SCALE = 100.0   # miss distance normaliser (units)
TIME_SCALE = 60.0    # time-to-closest-approach normaliser (ticks)
THREAT_MISS = 200.0  # ignore projectiles that will pass further than this


def _rects(obstacles, n):
    o = np.asarray(obstacles, dtype=np.float64)
    if o.ndim == 2:
        o = np.broadcast_to(o.reshape(1, -1, 4), (n, o.shape[0], 4))
    return o


def _slab_hits(ox, oy, ix, iy, rects):
    """Ray/segment slab test. ox, oy: (n,1,1) origins; ix, iy: inverse directions broadcastable to
    (n,m,1); rects (n,K,4). Returns t_enter, t_exit (n,m,K) in units of the direction vector."""
    x0, y0, x1, y1 = (rects[:, None, :, j] for j in range(4))
    tx1, tx2 = (x0 - ox) * ix, (x1 - ox) * ix
    ty1, ty2 = (y0 - oy) * iy, (y1 - oy) * iy
    te = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tx = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    return te, tx


def _safe_inv(v):
    return 1.0 / np.where(np.abs(v) < 1e-9, 1e-9, v)


_RAY_INV = _safe_inv(RAY_DIRS).astype(np.float32)


def ray_distances(me, obstacles):
    """Distance from me (n,2) to the nearest wall/obstacle along each of 8 rays -> (n,8)."""
    dx, dy = RAY_DIRS[:, 0], RAY_DIRS[:, 1]
    x, y = me[:, :1], me[:, 1:]
    with np.errstate(divide="ignore"):
        tx = np.where(dx > 1e-9, (C.W - x) / dx, np.where(dx < -1e-9, -x / dx, np.inf))
        ty = np.where(dy > 1e-9, (C.H - y) / dy, np.where(dy < -1e-9, -y / dy, np.inf))
    dist = np.minimum(tx, ty)
    rects = _rects(obstacles, me.shape[0])
    if rects.shape[1]:
        m = me.astype(np.float32)
        te, tx_ = _slab_hits(m[:, 0, None, None], m[:, 1, None, None],
                             _RAY_INV[None, :, 0, None], _RAY_INV[None, :, 1, None], rects.astype(np.float32))
        hit = (te <= tx_) & (tx_ >= 0)
        dist = np.minimum(dist, np.where(hit, np.maximum(te, 0), np.inf).min(-1))
    return dist


def line_of_sight(a, b, obstacles):
    rects = _rects(obstacles, a.shape[0])
    if not rects.shape[1]:
        return np.ones(a.shape[0], dtype=bool)
    inv = _safe_inv(b - a)
    te, tx = _slab_hits(a[:, 0, None, None], a[:, 1, None, None], inv[:, 0, None, None], inv[:, 1, None, None], rects)
    blocked = (te <= tx) & (tx >= 0) & (te <= 1)
    return ~blocked.any(-1)[:, 0]


def incoming_projectiles(ar, i):
    """Features of the K_PROJ most imminent enemy projectiles heading towards agent i."""
    n = ar.n
    me = ar.pos[:, i]
    pp, pv, pa = ar.ppos[:, 1 - i], ar.pvel[:, 1 - i], ar.pact[:, 1 - i]
    rel = pp - me[:, None, :]                                   # projectile relative to me
    v2 = np.maximum((pv * pv).sum(-1), 1e-9)
    t_star = -(rel * pv).sum(-1) / v2                           # ticks to closest approach
    miss = np.linalg.norm(rel + pv * t_star[..., None], axis=-1)
    key = np.where(pa & (t_star > 0) & (miss < THREAT_MISS), t_star, np.inf)
    order = np.argsort(key, axis=1)[:, :K_PROJ]
    take = lambda a: np.take_along_axis(a, order, 1)
    key_s = take(key)
    present = np.isfinite(key_s)
    rel_s = np.take_along_axis(rel, order[..., None], 1)
    vel_s = np.take_along_axis(pv, order[..., None], 1)
    feats = np.stack([
        rel_s[..., 0] / RW, rel_s[..., 1] / RH,
        vel_s[..., 0] / C.PROJ_SPEED, vel_s[..., 1] / C.PROJ_SPEED,
        np.minimum(take(miss) / MISS_SCALE, 2.0),
        np.minimum(np.where(present, key_s, 0) / TIME_SCALE, 2.0),
        present.astype(np.float64),
    ], -1) * present[..., None]
    return feats.reshape(n, K_PROJ * PROJ_FEATS)


def _weapon_onehot(w):
    return np.stack([w == 1, w == 2, w == 3], -1).astype(np.float32)


def nearest_items(ar, i):
    n = ar.n
    out = np.zeros((n, K_ITEMS, ITEM_FEATS), dtype=np.float32)
    iact = getattr(ar, "iact", None)
    if iact is None or not iact.any():
        return out.reshape(n, -1)
    rel = ar.ipos - ar.pos[:, i][:, None, :]
    dist = np.where(iact, np.linalg.norm(rel, axis=-1), np.inf)
    order = np.argsort(dist, axis=1)[:, :K_ITEMS]
    d = np.take_along_axis(dist, order, 1)
    present = np.isfinite(d)
    r = np.take_along_axis(rel, order[..., None], 1)
    ty = np.take_along_axis(ar.itype, order, 1)
    out[..., 0] = np.clip(r[..., 0] / RW, -3, 3)
    out[..., 1] = np.clip(r[..., 1] / RH, -3, 3)
    out[..., 2] = np.minimum(np.where(present, d, 0) / DIAG, 3)
    for k in range(5):
        out[..., 3 + k] = ty == k
    out[..., 8] = 1
    out *= present[..., None]
    return out.reshape(n, -1)


def encode(ar, i):
    """Observation for agent i of every env in arena ar -> (n, 94) float32."""
    me, op = ar.pos[:, i], ar.pos[:, 1 - i]
    d = op - me
    dist = np.linalg.norm(d, axis=-1)
    u = d / np.maximum(dist, 1e-9)[:, None]
    out = np.zeros((ar.n, OBS_DIM), dtype=np.float32)
    out[:, 0] = me[:, 0] / C.W * 2 - 1
    out[:, 1] = me[:, 1] / C.H * 2 - 1
    out[:, 2:4] = ar.vel[:, i] / C.AGENT_SPEED
    out[:, 4] = ar.hp[:, i] / C.MAX_HP
    out[:, 5] = np.minimum(ar.cd[:, i] / C.COOLDOWN, 2.5)
    out[:, 6] = np.clip(d[:, 0] / RW, -3, 3)
    out[:, 7] = np.clip(d[:, 1] / RH, -3, 3)
    out[:, 8] = np.minimum(dist / DIAG, 3)
    out[:, 9:11] = u
    out[:, 11:13] = ar.vel[:, 1 - i] / C.AGENT_SPEED
    out[:, 13] = ar.hp[:, 1 - i] / C.MAX_HP
    out[:, 14] = np.minimum(ar.cd[:, 1 - i] / C.COOLDOWN, 2.5)
    obst = ar.obstacles[..., :getattr(ar, "k_used", ar.obstacles.shape[-2]), :]
    out[:, 15] = line_of_sight(me, op, obst)
    out[:, 16:24] = np.minimum(ray_distances(me, obst) / DIAG, 1.0)
    out[:, 24:66] = incoming_projectiles(ar, i)
    out[:, 66] = 1 - ar.t / C.MAX_TICKS
    if hasattr(ar, "weapon"):
        w_me, w_op = ar.weapon[:, i], ar.weapon[:, 1 - i]
        out[:, 67:70] = _weapon_onehot(w_me)
        out[:, 70] = np.where(w_me > 0, ar.ammo[:, i] / _W_AMMO[w_me], 0)
        out[:, 71] = ar.shield[:, i] / C.SHIELD_MAX
        out[:, 72:75] = _weapon_onehot(w_op)
        out[:, 75] = ar.shield[:, 1 - i] / C.SHIELD_MAX
        out[:, 76:94] = nearest_items(ar, i)
    return out


def decode_action(ar, i, move_idx, aim_idx, shoot):
    """Discrete action indices for agent i -> (move vec (n,2), absolute aim (n,), shoot (n,))."""
    d = ar.pos[:, 1 - i] - ar.pos[:, i]
    base = np.arctan2(d[:, 1], d[:, 0])
    return MOVE_DIRS[move_idx], base + AIM_OFFSETS[aim_idx], np.asarray(shoot).astype(bool)
