"""Arena layouts for the 1920 x 1080 world. Every map is point-symmetric around the centre, so
neither spawn side is favoured. Rects are (x0, y0, x1, y1) in world units."""
import numpy as np

import config as C

CX, CY = C.W / 2, C.H / 2


def mirror(rects):
    """Add the point-reflection of every rect (skipping ones already centred)."""
    out = []
    for x0, y0, x1, y1 in rects:
        out.append((x0, y0, x1, y1))
        m = (C.W - x1, C.H - y1, C.W - x0, C.H - y0)
        if not np.allclose(m, (x0, y0, x1, y1)):
            out.append(m)
    return out


def _pillars():
    s = 64
    return [(x - s / 2, y - s / 2, x + s / 2, y + s / 2) for x in (320, 640, 960, 1280, 1600) for y in (270, 540, 810)
            if not (x == 960 and y == 540)]


MAPS = {
    "open": dict(label="Nyílt tér", rects=[]),
    "pillars": dict(label="Oszlopcsarnok", rects=_pillars()),
    "cross": dict(label="Kereszt", rects=[(930, 330, 990, 750), (700, 510, 1220, 570)] + mirror([
        (200, 160, 500, 188), (200, 188, 228, 400), (1420, 160, 1720, 188), (1692, 188, 1720, 400)])),
    "bunkers": dict(label="Bunkerek", rects=mirror([
        (300, 300, 500, 326), (300, 326, 326, 470), (474, 326, 500, 470),       # U-bunker
        (760, 120, 1160, 146), (1300, 250, 1330, 480),
        (820, 420, 900, 470)])),
    "corridors": dict(label="Folyosók", rects=[(930, 440, 990, 640)] + mirror([
        (80, 330, 820, 356), (1100, 330, 1840, 356), (330, 520, 390, 560), (1300, 180, 1360, 240)])),
    "maze": dict(label="Labirintus", rects=mirror([
        (200, 150, 620, 176), (200, 176, 226, 430), (760, 250, 786, 560), (420, 420, 640, 446),
        (1000, 110, 1026, 330), (1240, 290, 1540, 316)])),
}
DESIGNED = [k for k in MAPS if k != "open"]


def random_map(rng, n_pairs=None):
    """Random symmetric cover: 5-10 mirrored bars/blocks, keeping a clear ring along the walls."""
    n_pairs = rng.integers(5, 11) if n_pairs is None else n_pairs
    rects = []
    for _ in range(n_pairs):
        if rng.random() < 0.55:   # bar
            long, thick = rng.uniform(120, 420), rng.uniform(22, 34)
            w, h = (long, thick) if rng.random() < 0.5 else (thick, long)
        else:                     # block
            w, h = rng.uniform(50, 130), rng.uniform(50, 130)
        x0 = rng.uniform(90, C.W - 90 - w)
        y0 = rng.uniform(90, C.H - 90 - h)
        rects.append((x0, y0, x0 + w, y0 + h))
    return mirror(rects)[:C.MAX_OBST]


def sample_training_map(rng):
    """Map distribution used during training: 20% open, 35% designed, 45% random."""
    r = rng.random()
    if r < 0.20:
        return []
    if r < 0.55:
        return MAPS[DESIGNED[rng.integers(len(DESIGNED))]]["rects"]
    return random_map(rng)


for _k, _m in MAPS.items():
    assert len(_m["rects"]) <= C.MAX_OBST, _k
    for _r in _m["rects"]:
        assert 0 <= _r[0] < _r[2] <= C.W and 0 <= _r[1] < _r[3] <= C.H, (_k, _r)
