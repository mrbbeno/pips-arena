"""Head-to-head evaluation: trained policy vs random and rule-based baselines.

    python evaluate.py --ckpt runs/main/best.pt --games 1000
"""
import argparse
import time

import numpy as np

import config as C
from bots import PolicyAgent, StraferAgent, random_agent, rule_agent
from engine import Arena
from model import get_device, load_actor

THREAT_TICKS = 30  # a shot "on course" within 0.5 s counts as a threat for dodge stats


def run_matches(agent_a, agent_b, n_games=1000, seed=None, maps=None, items=False):
    """Agent A plays as index 0, B as index 1; each env plays exactly one round.
    maps: None = open arena, or an rng -> rects function (e.g. maps.sample_training_map).
    Returns outcome + behaviour stats from A's perspective."""
    ar = Arena(n_games, seed=seed, map_fn=maps, items=items)
    outcome = np.full(n_games, -2)          # -2 = still running
    length = np.zeros(n_games)
    P = C.MAX_PROJ
    threatened = np.zeros((n_games, P), dtype=bool)   # B's shots that were ever on course for A
    n_threat = n_threat_hit = 0
    lateral = moving = 0.0
    while (outcome == -2).any():
        live = outcome == -2
        ma, aa, sa = agent_a(ar, 0)
        mb, ab, sb = agent_b(ar, 1)
        # --- threat tracking on B's projectiles (before the step moves them)
        rel = ar.ppos[:, 1] - ar.pos[:, 0, None, :]
        pv = ar.pvel[:, 1]
        t_star = -(rel * pv).sum(-1) / np.maximum((pv * pv).sum(-1), 1e-9)
        miss = np.linalg.norm(rel + pv * t_star[..., None], axis=-1)
        on_course = ar.pact[:, 1] & (t_star > 0) & (t_star < THREAT_TICKS) & (miss < C.AGENT_R + C.PROJ_R)
        threatened |= on_course & live[:, None]
        was_active = ar.pact[:, 1].copy()

        res = ar.step(np.stack([ma, mb], 1), np.stack([aa, ab], 1), np.stack([sa, sb], 1))

        ended = was_active & (~ar.pact[:, 1] | res.fired[:, 1]) | res.proj_hit[:, 1]
        n_threat += (threatened & ended).sum()
        n_threat_hit += (threatened & res.proj_hit[:, 1]).sum()
        threatened &= ~ended
        threatened[res.done] = False  # shots still in flight when a round ends don't count
        # --- lateral (strafing) movement of A relative to line to B
        d = ar.pos[:, 1] - ar.pos[:, 0]
        u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        v = ar.vel[:, 0]
        speed = np.linalg.norm(v, axis=-1)
        lat = np.abs(v[:, 0] * u[:, 1] - v[:, 1] * u[:, 0])
        mv = live & (speed > 0.5)
        lateral += lat[mv].sum()
        moving += speed[mv].sum()

        fin = live & res.done
        outcome[fin] = res.winner[fin]
        length[fin] = ar.t[fin]
        if res.done.any():
            ar.reset(res.done)
    win, loss, draw = outcome == 0, outcome == 1, outcome == -1
    secs = length / C.FPS
    return {
        "games": n_games,
        "win": win.mean(), "draw": draw.mean(), "loss": loss.mean(),
        "t_win": secs[win].mean() if win.any() else float("nan"),
        "t_loss": secs[loss].mean() if loss.any() else float("nan"),
        "dodge": 1 - n_threat_hit / max(n_threat, 1),
        "threats": int(n_threat),
        "lateral": lateral / max(moving, 1e-9),
    }


def fmt(r):
    return (f"win {r['win']:6.1%}  draw {r['draw']:6.1%}  loss {r['loss']:6.1%}  "
            f"t_win {r['t_win']:5.1f}s  t_loss {r['t_loss']:5.1f}s  "
            f"dodge {r['dodge']:5.1%} ({r['threats']} threats)  lateral {r['lateral']:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/best.pt")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--maps", action="store_true", help="training map mix (cover) + pickups instead of the open arena")
    args = p.parse_args()
    from maps import sample_training_map
    kw = dict(maps=sample_training_map, items=True) if args.maps else {}
    dev = get_device()
    actor, meta = load_actor(args.ckpt, dev)
    ai = PolicyAgent(actor, dev, deterministic=not args.stochastic)
    print(f"checkpoint {args.ckpt}  {meta}  device {dev}")
    for name, opp in [("random", random_agent), ("rule", rule_agent), ("strafer", StraferAgent(seed=args.seed))]:
        t0 = time.time()
        r = run_matches(ai, opp, args.games, seed=args.seed, **kw)
        print(f"AI      vs {name:7s}: {fmt(r)}   [{time.time() - t0:.1f}s]")
    # reference: how the baselines themselves behave
    for name, a, b in [("rule    vs random ", rule_agent, random_agent),
                       ("strafer vs rule   ", StraferAgent(seed=1), rule_agent),
                       ("strafer vs strafer", StraferAgent(seed=1), StraferAgent(seed=2))]:
        print(f"{name}: {fmt(run_matches(a, b, args.games, seed=args.seed, **kw))}")


if __name__ == "__main__":
    main()
