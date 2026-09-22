"""Measure every difficulty preset (world.DIFFICULTIES) on the map mix with pickups.

    python calibrate.py --games 300
"""
import argparse

from bots import StraferAgent, rule_agent
from evaluate import run_matches
from maps import sample_training_map
from world import DIFFICULTIES, AIController


def agent(key, seed):
    c = AIController.from_difficulty(key, seed=seed)
    return lambda ar, i: c.act(ar, i)[:3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    kw = dict(maps=sample_training_map, items=True, seed=args.seed)
    print(f"{'level':11s} {'vs rule (win/draw/loss)':>26s}   {'vs strafer (win/draw/loss)':>28s}")
    for key in DIFFICULTIES:
        r = run_matches(agent(key, 1), rule_agent, args.games, **kw)
        s = run_matches(agent(key, 2), StraferAgent(seed=3), args.games, **kw)
        print(f"{key:11s} {r['win']:7.1%} {r['draw']:7.1%} {r['loss']:7.1%}      "
              f"{s['win']:7.1%} {s['draw']:7.1%} {s['loss']:7.1%}", flush=True)


if __name__ == "__main__":
    main()
