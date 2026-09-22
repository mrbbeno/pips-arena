"""Baseline opponents and the policy wrapper. An agent is a callable
(arena, i) -> (move (n,2), aim (n,), shoot (n,)) acting for agent index i in every env."""
import numpy as np
import torch

import config as C

from encoding import MOVE_DIRS, encode, decode_action

_rng = np.random.default_rng()


def random_agent(ar, i):
    """Uniformly random action every tick."""
    move = MOVE_DIRS[_rng.integers(0, len(MOVE_DIRS), ar.n)]
    return move, _rng.uniform(-np.pi, np.pi, ar.n), _rng.random(ar.n) < 0.5


def rule_agent(ar, i):
    """Walk straight at the opponent, aim straight at it, shoot whenever off cooldown."""
    d = ar.pos[:, 1 - i] - ar.pos[:, i]
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(dist, 1e-9), np.arctan2(d[:, 1], d[:, 0]), np.ones(ar.n, dtype=bool)


def lead_angle(shooter, target, target_vel, speed=C.PROJ_SPEED):
    """Aim angle that intercepts a target moving at constant velocity (falls back to direct aim)."""
    p = target - shooter
    a = (target_vel * target_vel).sum(-1) - speed ** 2
    b = 2 * (p * target_vel).sum(-1)
    c = (p * p).sum(-1)
    disc = np.maximum(b * b - 4 * a * c, 0)
    t = (-b - np.sqrt(disc)) / np.minimum(2 * a, -1e-9)  # a < 0 since target is slower than shots
    aim = p + target_vel * np.maximum(t, 0)[:, None]
    return np.arctan2(aim[:, 1], aim[:, 0])


class StraferAgent:
    """Harder scripted baseline: keeps a mid-range distance, circle-strafes (randomly
    reversing, bouncing off walls), leads its shots, and side-steps projectiles on a
    collision course."""

    def __init__(self, preferred=250, band=60, dodge_ticks=25, flip_prob=1 / 90, seed=None):
        self.preferred, self.band, self.dodge_ticks, self.flip_prob = preferred, band, dodge_ticks, flip_prob
        self.rng = np.random.default_rng(seed)
        self.orbit = None

    def __call__(self, ar, i):
        n = ar.n
        if self.orbit is None or self.orbit.shape[0] != n:
            self.orbit = self.rng.choice([-1.0, 1.0], n)
        self.orbit[ar.t == 0] = self.rng.choice([-1.0, 1.0], (ar.t == 0).sum())
        self.orbit[self.rng.random(n) < self.flip_prob] *= -1

        me, op = ar.pos[:, i], ar.pos[:, 1 - i]
        d = op - me
        dist = np.linalg.norm(d, axis=-1, keepdims=True)
        u = d / np.maximum(dist, 1e-9)
        tangent = np.stack([-u[:, 1], u[:, 0]], -1) * self.orbit[:, None]
        radial = np.where(dist > self.preferred + self.band, 1.0,
                          np.where(dist < self.preferred - self.band, -1.0, 0.0))
        move = tangent + 0.8 * radial * u

        # walls: reverse orbit when the strafe direction runs into a wall, and push off it
        margin = 70
        lo = me < margin
        hi = me > np.array([C.W, C.H]) - margin
        wall_push = lo.astype(float) - hi.astype(float)
        into_wall = ((tangent * wall_push).sum(-1) < -0.3)
        self.orbit[into_wall] *= -1
        move += 1.5 * wall_push

        # dodge: most imminent enemy shot that will hit if we stand still
        rel = ar.ppos[:, 1 - i] - me[:, None, :]
        pv = ar.pvel[:, 1 - i]
        t_star = -(rel * pv).sum(-1) / np.maximum((pv * pv).sum(-1), 1e-9)
        closest = rel + pv * t_star[..., None]
        miss = np.linalg.norm(closest, axis=-1)
        threat = ar.pact[:, 1 - i] & (t_star > 0) & (t_star < self.dodge_ticks) & (miss < C.AGENT_R + C.PROJ_R + 8)
        key = np.where(threat, t_star, np.inf)
        k = key.argmin(1)
        has = np.isfinite(key[np.arange(n), k])
        if has.any():
            e = np.flatnonzero(has)
            v = pv[e, k[e]] / np.maximum(np.linalg.norm(pv[e, k[e]], axis=-1, keepdims=True), 1e-9)
            perp = np.stack([-v[:, 1], v[:, 0]], -1)
            # step away from the shot's line: opposite side of the closest-approach point
            side = -np.sign((closest[e, k[e]] * perp).sum(-1))
            side[side == 0] = self.orbit[e][side == 0]
            dodge = perp * side[:, None]
            # if that side is blocked by a wall, go the other way
            blocked = (dodge * wall_push[e]).sum(-1) < -0.3
            dodge[blocked] *= -1
            move[e] = dodge + 0.5 * wall_push[e]

        return move, lead_angle(me, op, ar.vel[:, 1 - i]), np.ones(n, dtype=bool)


class PolicyAgent:
    def __init__(self, actor, device, deterministic=True):
        self.actor, self.device, self.deterministic = actor, device, deterministic

    @torch.no_grad()
    def __call__(self, ar, i):
        obs = torch.as_tensor(encode(ar, i), device=self.device)
        a, _ = self.actor.act(obs, self.deterministic)
        a = a.cpu().numpy()
        return decode_action(ar, i, a[:, 0], a[:, 1], a[:, 2])
