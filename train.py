"""Self-play PPO training.

    python train.py --minutes 3 --run smoke      # pipeline check
    python train.py --minutes 40 --run main      # real run
    python train.py --minutes 45 --run v2 --warm-start runs/main/best.pt   # maps + pickups, from the v1 AI

Writes runs/<run>/log.csv, checkpoints (ckpt_XXXX.pt, latest.pt, final.pt) and reward_curve.png.
"""
import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch

import config as C
from bots import PolicyAgent, StraferAgent, random_agent, rule_agent
from encoding import OBS_DIM, encode, decode_action
from engine import Arena
from evaluate import run_matches
from maps import sample_training_map
from model import Actor, Critic, get_device, load_actor, save_actor, widen_actor


def compute_rewards(res):
    """(n, 2) per-agent rewards for one tick."""
    r = C.HIT_REWARD * (res.hits - res.taken).astype(np.float64) - C.TICK_PENALTY
    r += C.PICKUP_REWARD * (res.picked > 0)
    r[res.done & (res.winner == -1)] -= C.DRAW_PENALTY
    for k in (0, 1):
        r[res.winner == k, k] += C.WIN_REWARD
        r[res.winner == k, 1 - k] -= C.WIN_REWARD
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=40)
    p.add_argument("--envs", type=int, default=C.N_ENVS)
    p.add_argument("--rollout", type=int, default=C.ROLLOUT)
    p.add_argument("--run", default="main")
    p.add_argument("--eval-every", type=int, default=15, help="iterations between baseline evals")
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-maps", action="store_true", help="open arena only (v1 setting)")
    p.add_argument("--no-items", action="store_true", help="no pickups (v1 setting)")
    p.add_argument("--warm-start", default="", help="actor checkpoint to start from (v1 nets are widened)")
    p.add_argument("--critic-warmup", type=int, default=0, help="iterations training only the critic")
    p.add_argument("--ent-start", type=float, default=C.ENT_START)
    p.add_argument("--ent-end", type=float, default=C.ENT_END)
    p.add_argument("--lr", type=float, default=C.LR)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = get_device()
    out = os.path.join("runs", args.run)
    os.makedirs(out, exist_ok=True)
    E, T = args.envs, args.rollout
    N = 2 * E  # columns: [agent 0 of every env, agent 1 of every env]
    n_pool = int(E * C.POOL_FRAC)
    print(f"device={dev}  envs={E}  rollout={T}  samples/iter={N * T}  budget={args.minutes} min")

    map_fn = None if args.no_maps else sample_training_map
    items = not args.no_items
    ar = Arena(E, seed=args.seed, map_fn=map_fn, items=items)
    actor, critic = Actor().to(dev), Critic().to(dev)
    ref = None
    if args.warm_start:
        base, _ = load_actor(args.warm_start, torch.device("cpu"))
        actor.load_state_dict(widen_actor(base).state_dict())
        ref = base.to(dev).eval()     # frozen original, for head-to-head evals
        print(f"warm start from {args.warm_start} (obs {base.obs_dim} -> {actor.obs_dim})")
    opp = Actor().to(dev).eval()
    params = [*actor.parameters(), *critic.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    pool = []
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"actor params: {n_params:,}")

    b_obs = torch.zeros(T, N, OBS_DIM, device=dev)
    b_act = torch.zeros(T, N, 3, dtype=torch.long, device=dev)
    b_logp = torch.zeros(T, N, device=dev)
    b_val = torch.zeros(T, N, device=dev)
    b_rew = torch.zeros(T, N, device=dev)
    b_done = torch.zeros(T, N, device=dev)

    def observe():
        return torch.as_tensor(np.concatenate([encode(ar, 0), encode(ar, 1)]), device=dev)

    ep_ret = np.zeros((E, 2))
    ep_hits = np.zeros(E)
    log_f = open(os.path.join(out, "log.csv"), "w", newline="")
    log = None
    obs = observe()
    start = time.time()
    agent_steps = 0
    it = 0
    last_eval = {}
    while (time.time() - start) / 60 < args.minutes:
        it += 1
        frac = min((time.time() - start) / 60 / args.minutes, 1.0)
        ent_coef = args.ent_start + (args.ent_end - args.ent_start) * frac
        warmup = it <= args.critic_warmup
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - 0.8 * frac)
        use_pool = bool(pool) and n_pool > 0
        if use_pool:
            opp.load_state_dict(random.choice(pool))

        # ------------------------------------------------------------ rollout
        stats = {"len": [], "winner": [], "hits": [], "pool_win": [], "pool_loss": []}
        t_roll = time.time()
        actor.eval()
        for t in range(T):
            with torch.no_grad():
                a, logp = actor.act(obs)
                v = critic(obs)
                if use_pool:
                    a[E:E + n_pool], _ = opp.act(obs[E:E + n_pool])
            b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, a, logp, v
            an = a.cpu().numpy()
            m0, aim0, s0 = decode_action(ar, 0, an[:E, 0], an[:E, 1], an[:E, 2])
            m1, aim1, s1 = decode_action(ar, 1, an[E:, 0], an[E:, 1], an[E:, 2])
            res = ar.step(np.stack([m0, m1], 1), np.stack([aim0, aim1], 1), np.stack([s0, s1], 1))
            r = compute_rewards(res)
            ep_ret += r
            ep_hits += res.hits.sum(1)
            done = res.done
            if done.any():
                idx = np.flatnonzero(done)
                stats["len"] += list(ar.t[idx] / C.FPS)
                stats["winner"] += list(res.winner[idx])
                stats["hits"] += list(ep_hits[idx])
                pidx = idx[idx < n_pool] if use_pool else idx[:0]
                stats["pool_win"] += list(res.winner[pidx] == 0)
                stats["pool_loss"] += list(res.winner[pidx] == 1)
                ep_ret[idx] = 0
                ep_hits[idx] = 0
                ar.reset(done)
            b_rew[t] = torch.as_tensor(np.concatenate([r[:, 0], r[:, 1]]), device=dev)
            b_done[t] = torch.as_tensor(np.concatenate([done, done]), device=dev, dtype=torch.float32)
            obs = observe()
        agent_steps += N * T
        t_roll = time.time() - t_roll

        # --------------------------------------------------------------- GAE
        t_upd = time.time()
        with torch.no_grad():
            next_v = critic(obs)
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + C.GAMMA * nv * nonterm - b_val[t]
                last = delta + C.GAMMA * C.LAMBDA * nonterm * last
                adv[t] = last
            ret = adv + b_val

        # columns played by the frozen pool opponent are not trained on
        keep = torch.ones(N, dtype=torch.bool, device=dev)
        if use_pool:
            keep[E:E + n_pool] = False
        f_obs, f_act, f_logp = b_obs[:, keep].reshape(-1, OBS_DIM), b_act[:, keep].reshape(-1, 3), b_logp[:, keep].reshape(-1)
        f_adv, f_ret = adv[:, keep].reshape(-1), ret[:, keep].reshape(-1)

        # ---------------------------------------------------------------- PPO
        actor.train()
        M = f_obs.shape[0]
        pg_l, v_l, ent_l, kl_l, clip_l = [], [], [], [], []
        for _ in range(C.EPOCHS):
            perm = torch.randperm(M, device=dev)
            for s in range(0, M, C.MINIBATCH):
                mb = perm[s:s + C.MINIBATCH]
                logp, ent = actor.evaluate(f_obs[mb], f_act[mb])
                ratio = torch.exp(logp - f_logp[mb])
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - C.CLIP, 1 + C.CLIP) * a_mb).mean()
                vloss = 0.5 * (critic(f_obs[mb]) - f_ret[mb]).pow(2).mean()
                loss = C.VF_COEF * vloss if warmup else pg + C.VF_COEF * vloss - ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, C.MAX_GRAD_NORM)
                opt.step()
                with torch.no_grad():
                    pg_l.append(pg.item()); v_l.append(vloss.item()); ent_l.append(ent.mean().item())
                    kl_l.append((f_logp[mb] - logp).mean().item())
                    clip_l.append(((ratio - 1).abs() > C.CLIP).float().mean().item())
        t_upd = time.time() - t_upd

        if it % C.POOL_EVERY == 0:
            pool.append({k: v.detach().clone() for k, v in actor.state_dict().items()})
            pool = pool[-C.POOL_SIZE:]

        # --------------------------------------------------------- eval / log
        if it % args.eval_every == 0 or it == 1:
            actor.eval()
            ai = PolicyAgent(actor, dev, deterministic=True)
            last_eval = {}
            opponents = [("rule", rule_agent), ("strafer", StraferAgent(seed=it))]
            if ref is not None:
                opponents.append(("v1", PolicyAgent(ref, dev, deterministic=True)))
            for name, bot in opponents:
                rr = run_matches(ai, bot, args.eval_games, seed=it, maps=map_fn, items=items)
                last_eval[f"{name}_win"] = rr["win"]
                last_eval[f"{name}_loss"] = rr["loss"]
                last_eval[f"{name}_dodge"] = rr["dodge"]
                if name == "rule":
                    last_eval["lateral"] = rr["lateral"]
            ev = last_eval
        else:
            ev = {}
        w = np.array(stats["winner"])
        row = {
            "iter": it,
            "minutes": round((time.time() - start) / 60, 3),
            "agent_steps": agent_steps,
            "sps": round(N * T / (t_roll + t_upd)),
            "episodes": len(w),
            "ep_len_s": round(float(np.mean(stats["len"])), 2) if len(w) else "",
            "draw_rate": round(float((w == -1).mean()), 3) if len(w) else "",
            "hits_per_ep": round(float(np.mean(stats["hits"])), 2) if len(w) else "",
            "pool_win": round(float(np.mean(stats["pool_win"])), 3) if stats["pool_win"] else "",
            "pool_loss": round(float(np.mean(stats["pool_loss"])), 3) if stats["pool_loss"] else "",
            "entropy": round(float(np.mean(ent_l)), 3),
            "pg_loss": round(float(np.mean(pg_l)), 4),
            "v_loss": round(float(np.mean(v_l)), 4),
            "kl": round(float(np.mean(kl_l)), 4),
            "clipfrac": round(float(np.mean(clip_l)), 3),
            "rule_win": "", "rule_loss": "", "rule_dodge": "",
            "random_win": "", "random_loss": "", "random_dodge": "", "lateral": "",
            "strafer_win": "", "strafer_loss": "", "strafer_dodge": "",
            "v1_win": "", "v1_loss": "", "v1_dodge": "",
        }
        row.update({k: round(float(v), 3) for k, v in ev.items()})
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        msg = (f"it {it:4d} {row['minutes']:6.2f}m steps {agent_steps / 1e6:6.2f}M sps {row['sps']:>7} "
               f"| len {row['ep_len_s']}s draw {row['draw_rate']} hits {row['hits_per_ep']} "
               f"pool_w {row['pool_win']} | ent {row['entropy']} kl {row['kl']} v {row['v_loss']}")
        if ev:
            msg += (f"\n      EVAL vs rule win {ev['rule_win']:.1%} loss {ev['rule_loss']:.1%} dodge {ev['rule_dodge']:.1%}"
                    + (f" | vs V1 win {ev['v1_win']:.1%} loss {ev['v1_loss']:.1%}" if "v1_win" in ev else "")
                    + f" | vs STRAFER win {ev['strafer_win']:.1%}"
                    f" loss {ev['strafer_loss']:.1%} dodge {ev['strafer_dodge']:.1%} | lateral {ev['lateral']:.2f}")
        print(msg, flush=True)

        if it % args.ckpt_every == 0:
            save_actor(os.path.join(out, f"ckpt_{it:04d}.pt"), actor, iter=it, agent_steps=agent_steps)
            save_actor(os.path.join(out, "latest.pt"), actor, iter=it, agent_steps=agent_steps)

    save_actor(os.path.join(out, "final.pt"), actor, iter=it, agent_steps=agent_steps)
    log_f.close()
    print(f"done: {it} iterations, {agent_steps / 1e6:.1f}M agent-steps in {(time.time() - start) / 60:.1f} min")
    try:
        from plot import plot_run
        plot_run(out)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
