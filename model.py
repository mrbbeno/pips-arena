"""Actor (policy) and critic MLPs. The actor alone is used at play time."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import config as C
from encoding import OBS_DIM, OBS_DIM_V1, N_MOVE, N_AIM


def get_device(pref=None):
    pref = C.DEVICE if pref is None else pref
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _layer(i, o, gain=np.sqrt(2)):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def _body(hidden, obs_dim=OBS_DIM):
    return nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh())


class Actor(nn.Module):
    """obs (94) -> logits for move (9), aim offset (7), shoot (2)."""

    def __init__(self, hidden=C.HIDDEN, obs_dim=OBS_DIM):
        super().__init__()
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.body = _body(hidden, obs_dim)
        self.move = _layer(hidden, N_MOVE, 0.01)
        self.aim = _layer(hidden, N_AIM, 0.01)
        self.shoot = _layer(hidden, 2, 0.01)

    def forward(self, x):
        h = self.body(x[..., :self.obs_dim])
        return self.move(h), self.aim(h), self.shoot(h)

    def dists(self, x):
        return [Categorical(logits=l) for l in self(x)]

    def act(self, x, deterministic=False):
        ds = self.dists(x)
        a = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in ds], -1)
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(ds))
        return a, logp

    def evaluate(self, x, a):
        ds = self.dists(x)
        logp = sum(d.log_prob(a[:, k]) for k, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds)
        return logp, ent


class Critic(nn.Module):
    def __init__(self, hidden=C.HIDDEN):
        super().__init__()
        self.net = nn.Sequential(_body(hidden), _layer(hidden, 1, 1.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FastPolicy:
    """Play-time inference: the actor's weights copied to numpy. One observation in,
    argmax (or sampled) action out, with no torch/distribution overhead."""

    def __init__(self, actor, deterministic=True, seed=None):
        sd = {k: v.detach().float().cpu().numpy() for k, v in actor.state_dict().items()}
        self.w1, self.b1 = sd["body.0.weight"].T.copy(), sd["body.0.bias"]
        self.in_dim = self.w1.shape[0]   # v1 checkpoints take only the first 67 features
        self.w2, self.b2 = sd["body.2.weight"].T.copy(), sd["body.2.bias"]
        self.wh = np.concatenate([sd["move.weight"], sd["aim.weight"], sd["shoot.weight"]]).T.copy()
        self.bh = np.concatenate([sd["move.bias"], sd["aim.bias"], sd["shoot.bias"]])
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def logits(self, obs):
        h = np.tanh(obs[..., :self.in_dim] @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        z = h @ self.wh + self.bh
        return z[..., :N_MOVE], z[..., N_MOVE:N_MOVE + N_AIM], z[..., N_MOVE + N_AIM:]

    def _pick(self, z):
        if self.deterministic:
            return z.argmax(-1)
        p = np.exp(z - z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return (p.cumsum(-1) > self.rng.random(p.shape[:-1] + (1,))).argmax(-1)

    def decide(self, obs):
        """obs (n, 67) float32 -> move_idx, aim_idx, shoot (n,), shoot probability (n,)."""
        zm, za, zs = self.logits(obs)
        p_shoot = 1.0 / (1.0 + np.exp(zs[..., 0] - zs[..., 1]))
        return self._pick(zm), self._pick(za), self._pick(zs), p_shoot


def save_actor(path, actor, **meta):
    torch.save({"state_dict": actor.state_dict(), "hidden": actor.hidden, "obs_dim": actor.obs_dim,
                "meta": meta}, path)


def widen_actor(actor, obs_dim=OBS_DIM):
    """Copy of `actor` accepting obs_dim inputs; the new input columns get zero weights, so the
    widened network initially behaves exactly like the original (warm start for v2 features)."""
    wide = Actor(actor.hidden, obs_dim)
    sd = {k: v.clone() for k, v in actor.state_dict().items()}
    w = sd["body.0.weight"]
    pad = torch.zeros(w.shape[0], obs_dim - w.shape[1], dtype=w.dtype)
    sd["body.0.weight"] = torch.cat([w.cpu(), pad], 1)
    wide.load_state_dict({k: v.cpu() for k, v in sd.items()})
    return wide


def load_actor(path, device=None):
    device = get_device() if device is None else device
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(ck["hidden"], ck.get("obs_dim", OBS_DIM_V1)).to(device)
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return actor, ck.get("meta", {})
