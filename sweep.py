"""Checkpoint sweep for a run: every k-th checkpoint vs the strafer bot and vs a reference model,
on the training map mix with pickups. Picks the best checkpoint and copies it to best.pt.

    python sweep.py --run v2 --ref runs/main/best.pt --every 50 --games 300
"""
import argparse
import glob
import os
import shutil

from bots import PolicyAgent, StraferAgent
from evaluate import fmt, run_matches
from maps import sample_training_map
from model import get_device, load_actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="v2")
    p.add_argument("--ref", default="runs/main/best.pt")
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    dev = get_device()
    ref_actor, _ = load_actor(args.ref, dev)
    ref = PolicyAgent(ref_actor, dev)
    paths = [q for q in sorted(glob.glob(f"runs/{args.run}/ckpt_*.pt")) if int(q[-7:-3]) % args.every == 0]
    paths.append(f"runs/{args.run}/final.pt")
    kw = dict(maps=sample_training_map, items=True, seed=args.seed)
    results = []
    for path in paths:
        actor, _ = load_actor(path, dev)
        ai = PolicyAgent(actor, dev)
        s = run_matches(ai, StraferAgent(seed=args.seed), args.games, **kw)
        v = run_matches(ai, ref, args.games, **kw)
        score = min(s["win"], v["win"] + 0.5 * v["draw"]) - 0.5 * v["draw"]
        results.append((score, path, s, v))
        print(f"{os.path.basename(path):14s} strafer: {fmt(s)}\n{'':14s} vs ref : {fmt(v)}", flush=True)
    best = max(results, key=lambda r: r[0])
    shutil.copy(best[1], f"runs/{args.run}/best.pt")
    print("BEST:", best[1], "-> copied to", f"runs/{args.run}/best.pt")


if __name__ == "__main__":
    main()
