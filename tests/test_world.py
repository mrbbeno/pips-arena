import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from encoding import encode  # noqa: E402
from world import World, duel_view, pair_view  # noqa: E402


def test_pair_view_encodes_like_the_engine():
    """The AI's view of a 1v1 world must equal the engine's own 2-agent encoding."""
    rng = np.random.default_rng(0)
    w = World([C.MAX_HP, C.MAX_HP], team=[0, 1], obstacles=[(350, 250, 450, 350)], items=True, seed=3)
    for _ in range(400):
        w.step(rng.uniform(-1, 1, (2, 2)), rng.uniform(-np.pi, np.pi, 2), rng.random(2) < 0.5)
        v = pair_view(w, [1], 0, t_cap=10 ** 9)
        assert np.allclose(encode(w.ar, 1), encode(v, 0), atol=1e-6)
        if not w.alive.all():
            break


def test_events_fire_hit_death():
    w = World([C.MAX_HP, 1], items=False)
    w.pos[:] = [[200, 300], [500, 300]]
    ev = w.step(np.zeros((2, 2)), np.array([0.0, np.pi]), np.array([True, False]))
    assert [f[0] for f in ev.fired] == [0]
    hits, deaths = [], []
    for _ in range(40):
        ev = w.step(np.zeros((2, 2)), np.array([0.0, np.pi]), np.array([False, False]))
        hits += ev.hits
        deaths += ev.deaths
    assert hits and hits[0][:2] == (0, 1) and deaths and deaths[0][0] == 1


def test_survival_enemy_shots_pass_through_other_enemies():
    w = World([10, 3, 3], items=False)
    w.pos[:] = [[100, 300], [300, 300], [500, 300]]
    aim = np.array([0.0, 0.0, np.pi])
    w.step(np.zeros((3, 2)), aim, np.array([False, False, True]))
    for _ in range(60):
        w.step(np.zeros((3, 2)), aim, np.zeros(3, bool))
    assert w.hp[1] == 3 and w.hp[0] == 9


def test_pickup_event_and_dodge_event():
    w = World([C.MAX_HP, C.MAX_HP], items=True, seed=1)
    w.ar.iact[0, 0] = True
    w.ar.ipos[0, 0] = w.pos[0]
    w.ar.itype[0, 0] = 1  # rapid
    ev = w.step(np.zeros((2, 2)), w.aim.copy(), np.zeros(2, bool))
    assert ev.pickups and ev.pickups[0][:2] == (0, 1) and w.weapon[0] == 2
    # a shot that passes 30 units from agent 1 and flies off the map is a dodge by agent 1
    w.pos[:] = [[100, 330], [500, 300]]
    dodges = []
    w.step(np.zeros((2, 2)), np.array([0.0, 0.0]), np.array([True, False]))
    for _ in range(int(C.W / C.PROJ_SPEED) + 5):
        dodges += w.step(np.zeros((2, 2)), np.array([0.0, 0.0]), np.zeros(2, bool)).dodges
    assert (1, 0) in dodges


def test_duel_view_rescales_hp_and_batches_enemies():
    w = World([10, 3, 3], items=False)
    w.hp[:] = [5, 3, 1]
    v = duel_view(w, [1, 2])
    assert v.n == 2 and v.hp.tolist() == [[5, 3], [2, 3]]
    assert encode(v, 0).shape == (2, 94)
