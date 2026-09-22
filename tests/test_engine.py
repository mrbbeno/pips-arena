import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from encoding import OBS_DIM, K_PROJ, PROJ_FEATS, encode, decode_action  # noqa: E402
from engine import Arena, circle_rect_dist  # noqa: E402

STILL = np.zeros((1, 2, 2))
NO_SHOT = np.zeros((1, 2), dtype=bool)


def make(p0, p1, obstacles=()):
    ar = Arena(1, obstacles=list(obstacles), seed=0)
    ar.pos[0, 0] = p0
    ar.pos[0, 1] = p1
    return ar


def aim_at(ar):
    d = ar.pos[0, 1] - ar.pos[0, 0]
    a = np.arctan2(d[1], d[0])
    return np.array([[a, a + np.pi]])


def run(ar, ticks, move=STILL, shoot=NO_SHOT, aim=None):
    res = None
    for _ in range(ticks):
        res = ar.step(move, aim_at(ar) if aim is None else aim, shoot)
        if res.done[0]:
            break
    return res


# ------------------------------------------------------------ projectile hits
def test_projectile_hits_opponent_and_costs_one_hp():
    ar = make([200, 300], [500, 300])
    total = 0
    res = ar.step(STILL, aim_at(ar), np.array([[True, False]]))
    for _ in range(40):
        res = ar.step(STILL, aim_at(ar), NO_SHOT)
        total += res.hits[0, 0]
    assert total == 1
    assert ar.hp[0, 1] == C.MAX_HP - 1
    assert ar.hp[0, 0] == C.MAX_HP
    assert not ar.pact.any(), "projectile must disappear on hit"


def test_projectile_aimed_away_misses():
    ar = make([200, 300], [500, 300])
    aim = np.array([[np.pi, 0.0]])  # agent 0 shoots left, away from agent 1
    ar.step(STILL, aim, np.array([[True, False]]))
    run(ar, 60, aim=aim)
    assert ar.hp[0, 1] == C.MAX_HP


def test_projectile_near_miss_does_not_hit():
    ar = make([200, 300], [500, 300 + C.AGENT_R + C.PROJ_R + 2])
    ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
    run(ar, 60, aim=np.array([[0.0, 0.0]]))
    assert ar.hp[0, 1] == C.MAX_HP


def test_projectile_grazing_hits():
    ar = make([200, 300], [500, 300 + C.AGENT_R + C.PROJ_R - 1])
    ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
    run(ar, 60, aim=np.array([[0.0, 0.0]]))
    assert ar.hp[0, 1] == C.MAX_HP - 1


def test_fast_projectile_does_not_tunnel():
    """Swept collision: sampling every position individually would still be caught."""
    for offset in np.linspace(0, C.PROJ_SPEED, 7):
        ar = make([200, 300], [400 + offset, 300])
        ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
        run(ar, 60, aim=np.array([[0.0, 0.0]]))
        assert ar.hp[0, 1] == C.MAX_HP - 1


def test_projectile_removed_at_arena_edge():
    ar = make([100, 300], [100, 100])
    ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
    assert ar.pact[0, 0].sum() == 1
    run(ar, int(C.W / C.PROJ_SPEED) + 5, aim=np.array([[0.0, 0.0]]))
    assert not ar.pact.any()


def test_projectile_blocked_by_obstacle():
    ar = make([200, 300], [600, 300], obstacles=[(380, 250, 420, 350)])
    ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
    run(ar, 80, aim=np.array([[0.0, 0.0]]))
    assert ar.hp[0, 1] == C.MAX_HP
    assert not ar.pact.any()


def test_own_projectile_never_hits_shooter():
    ar = make([200, 300], [600, 100])
    ar.step(STILL, np.array([[0.0, 0.0]]), np.array([[True, False]]))
    # shooter runs after its own bullet
    run(ar, 60, move=np.array([[[1, 0], [0, 0]]]), aim=np.array([[0.0, 0.0]]))
    assert ar.hp[0, 0] == C.MAX_HP


# ----------------------------------------------------------------- cooldown
def test_cooldown_limits_fire_rate():
    ar = make([200, 300], [600, 100])
    fired = 0
    for _ in range(C.COOLDOWN * 3):
        before = ar.pact.sum()
        ar.step(STILL, np.array([[np.pi / 2, 0]]), np.array([[True, False]]))
        fired += ar.pact.sum() > before
    assert fired == 3


# ------------------------------------------------------------- agent vs wall
def test_agent_stays_inside_arena():
    ar = make([30, 30], [600, 400])
    run(ar, 100, move=np.array([[[-1, -1], [1, 1]]]))
    assert ar.pos[0, 0, 0] >= C.AGENT_R - 1e-9 and ar.pos[0, 0, 1] >= C.AGENT_R - 1e-9
    assert ar.pos[0, 1, 0] <= C.W - C.AGENT_R + 1e-9 and ar.pos[0, 1, 1] <= C.H - C.AGENT_R + 1e-9


def test_agent_cannot_pass_through_obstacle():
    rect = (380, 200, 420, 400)
    ar = make([300, 300], [700, 550], obstacles=[rect])
    run(ar, 120, move=np.array([[[1, 0], [0, 0]]]))
    assert ar.pos[0, 0, 0] <= rect[0] - C.AGENT_R + 1e-6


def test_agents_do_not_overlap():
    ar = make([300, 300], [400, 300])
    run(ar, 60, move=np.array([[[1, 0], [-1, 0]]]))
    assert np.linalg.norm(ar.pos[0, 0] - ar.pos[0, 1]) >= 2 * C.AGENT_R - 1e-6


# ------------------------------------------------------------ win conditions
def test_win_when_opponent_hp_reaches_zero():
    ar = make([200, 300], [500, 300])
    ar.hp[0, 1] = 1
    res = run(ar, 60, shoot=np.array([[True, False]]))
    assert res.done[0] and res.winner[0] == 0


def test_agent1_can_win():
    ar = make([200, 300], [500, 300])
    ar.hp[0, 0] = 1
    res = run(ar, 60, shoot=np.array([[False, True]]))
    assert res.done[0] and res.winner[0] == 1


def test_simultaneous_death_is_draw():
    ar = make([200, 300], [500, 300])
    ar.hp[0] = 1
    res = run(ar, 60, shoot=np.array([[True, True]]))
    assert res.done[0] and res.winner[0] == -1


def test_timeout_is_draw():
    ar = make([200, 300], [500, 300])
    ar.t[0] = C.MAX_TICKS - 5
    res = run(ar, 10)
    assert res.done[0] and res.winner[0] == -1 and (ar.hp[0] == C.MAX_HP).all()


def test_not_done_mid_round():
    ar = make([200, 300], [500, 300])
    res = ar.step(STILL, aim_at(ar), NO_SHOT)
    assert not res.done[0] and res.winner[0] == -1


def test_reset_mask_only_resets_selected():
    ar = Arena(4, seed=1)
    ar.hp[:] = 2
    ar.reset(np.array([True, False, True, False]))
    assert (ar.hp[[0, 2]] == C.MAX_HP).all() and (ar.hp[[1, 3]] == 2).all()


def test_spawns_respect_min_distance():
    ar = Arena(500, seed=2)
    assert (np.linalg.norm(ar.pos[:, 0] - ar.pos[:, 1], axis=-1) >= C.SPAWN_MIN_DIST).all()


# ------------------------------------------------------------------ encoding
def test_encoding_shape_padding_and_symmetry():
    ar = make([200, 300], [500, 300])
    obs = encode(ar, 0)
    assert obs.shape == (1, OBS_DIM) and obs.dtype == np.float32
    assert (obs[0, 24:66] == 0).all(), "no projectiles -> zero padding"
    # agent 1 fires at agent 0: exactly one threat slot filled, on a collision course
    ar.step(STILL, aim_at(ar), np.array([[False, True]]))
    obs = encode(ar, 0)
    proj = obs[0, 24:66].reshape(K_PROJ, PROJ_FEATS)
    assert proj[0, 6] == 1 and (proj[1:] == 0).all()
    assert proj[0, 4] < 0.01  # predicted miss distance ~0
    assert np.isfinite(obs).all()
    # agent 1 sees nothing incoming
    assert (encode(ar, 1)[0, 24:66] == 0).all()


def test_decode_relative_aim():
    ar = make([200, 300], [200, 500])
    move, aim, shoot = decode_action(ar, 0, np.array([3]), np.array([3]), np.array([1]))
    assert np.allclose(aim, np.pi / 2) and shoot[0] and np.allclose(move, [[0, 1]])


# ------------------------------------------------------------------ model
def test_fast_policy_matches_torch_actor():
    import torch
    from model import Actor, FastPolicy
    torch.manual_seed(0)
    actor = Actor().eval()
    obs = np.random.default_rng(0).standard_normal((64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        a, _ = actor.act(torch.as_tensor(obs), deterministic=True)
    m, aim, s, p = FastPolicy(actor).decide(obs)
    assert (np.stack([m, aim, s], -1) == a.numpy()).all()
    assert ((p > 0.5) == (s == 1)).all()


# ------------------------------------------------------------------ v2: weapons, shields, pickups, maps
def fire_once(ar, weapon, aim=0.0):
    ar.weapon[0, 0] = weapon
    ar.ammo[0, 0] = 99
    ar.step(STILL, np.array([[aim, 0.0]]), np.array([[True, False]]))


def test_shotgun_fires_three_pellets_with_short_range():
    ar = make([100, 300], [700, 100])
    fire_once(ar, 1)
    assert ar.pact[0, 0].sum() == 3
    ang = np.degrees(np.arctan2(ar.pvel[0, 0, ar.pact[0, 0], 1], ar.pvel[0, 0, ar.pact[0, 0], 0]))
    assert np.allclose(sorted(ang), [-C.W_SPREAD_DEG[1], 0, C.W_SPREAD_DEG[1]])
    run(ar, C.W_LIFE[1] + 2, aim=np.array([[0.0, 0.0]]))
    assert not ar.pact.any(), "shotgun pellets fizzle after W_LIFE ticks"


def test_rail_does_two_damage_and_ammo_runs_out():
    ar = make([200, 300], [500, 300])
    ar.weapon[0, 0], ar.ammo[0, 0] = 3, 1
    ar.step(STILL, aim_at(ar), np.array([[True, False]]))
    assert ar.weapon[0, 0] == 0, "weapon reverts to blaster when ammo is spent"
    run(ar, 30)
    assert ar.hp[0, 1] == C.MAX_HP - 2


def test_shield_absorbs_before_hp():
    ar = make([200, 300], [500, 300])
    ar.shield[0, 1] = 1
    for _ in range(2):
        ar.step(STILL, aim_at(ar), np.array([[True, False]]))
        run(ar, C.COOLDOWN + 40)
    assert ar.shield[0, 1] == 0 and ar.hp[0, 1] == C.MAX_HP - 1


def test_items_spawn_away_from_walls_and_can_be_picked_up():
    ar = Arena(1, obstacles=[(300, 200, 500, 400)], items=True, seed=4)
    for _ in range(C.ITEM_FIRST + 2):
        ar.step(np.zeros((1, 2, 2)), ar.aim, np.zeros((1, 2), bool))
    assert ar.iact[0].sum() == 1
    p = ar.ipos[0, ar.iact[0]][0]
    assert circle_rect_dist(p, np.array([300, 200, 500, 400])) > C.ITEM_R
    ar.pos[0, 0] = p
    ar.itype[0, ar.iact[0]] = 3  # shield
    res = ar.step(np.zeros((1, 2, 2)), ar.aim, np.zeros((1, 2), bool))
    assert res.picked[0, 0] == 4 and ar.shield[0, 0] == C.SHIELD_ADD and not ar.iact[0].any()


def test_per_env_maps_and_blocked_line_of_sight():
    from encoding import line_of_sight
    ar = Arena(2, seed=0)
    ar.set_map(1, [(380, 100, 420, 500)])
    ar.pos[:, 0] = [200, 300]
    ar.pos[:, 1] = [600, 300]
    los = line_of_sight(ar.pos[:, 0], ar.pos[:, 1], ar.obstacles)
    assert los.tolist() == [True, False]
    obs = encode(ar, 0)
    assert obs[0, 15] == 1 and obs[1, 15] == 0
    assert obs[1, 16] < obs[0, 16], "ray to the right hits the wall in env 1"


def test_map_fn_spawns_clear_of_walls():
    from maps import sample_training_map
    ar = Arena(200, map_fn=sample_training_map, seed=1)
    for e in range(ar.n):
        for a in range(2):
            assert (circle_rect_dist(ar.pos[e, a], ar.obstacles[e]) > C.AGENT_R).all()


def test_team_winner_with_three_agents():
    ar = Arena(1, n_agents=3, max_hp=[5, 1, 1], team=[0, 1, 1], seed=0)
    ar.hp[0, 1] = 0
    ar.alive[0, 1] = False
    ar.pos[0] = [[200, 300], [700, 500], [500, 300]]
    ar.aim[0, 0] = 0.0
    res = None
    for _ in range(60):
        res = ar.step(np.zeros((1, 3, 2)), np.array([[0.0, 0.0, np.pi]]), np.array([[True, False, False]]))
        if res.done[0]:
            break
    assert res.done[0] and res.winner[0] == 0


def test_widened_v1_actor_behaves_identically():
    import torch
    from model import Actor, widen_actor
    torch.manual_seed(1)
    v1 = Actor(obs_dim=67).eval()
    v2 = widen_actor(v1).eval()
    x = torch.randn(32, OBS_DIM)
    with torch.no_grad():
        assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(v1(x), v2(x)))
