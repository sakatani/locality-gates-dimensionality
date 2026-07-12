"""Phase 13 — LGRC task: Latent-Grid Region Connectivity from an unordered set.

A latent W×W grid; a smooth low-frequency field defines a land/water mask
(connected 2D blobs); two land cells S,T are marked. Label = are S,T in the same
4-connected land component? The cells are handed as an UNORDERED SET with
features [u, v, land, is_S, is_T], where (u,v) is the true position affinely
scrambled by an unknown rotation/shear + noise (recoverable, not given). No true
(x,y), no adjacency → the model must recover a layout then flood-fill.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

N_FEAT = 5  # [u, v, land, is_S, is_T]


@dataclass(frozen=True)
class LGRCInstance:
    feats: np.ndarray      # (N, 5) unordered-set features
    label: int             # 1 = same land component
    W: int
    s_idx: int             # row index (into feats) of S
    t_idx: int             # row index of T
    true_rc: np.ndarray    # (N, 2) true (row,col) per set row — diagnostics only


def _smooth_field(W: int, rng: np.random.Generator, n_waves: int = 3) -> np.ndarray:
    ys, xs = np.mgrid[0:W, 0:W] / max(W - 1, 1)
    f = np.zeros((W, W))
    for _ in range(n_waves):
        fx, fy = rng.integers(1, 3), rng.integers(1, 3)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        f += amp * np.cos(2 * np.pi * (fx * xs + fy * ys) + ph)
    return f


def _components(land: np.ndarray) -> np.ndarray:
    """4-connected component id per cell; -1 for water."""
    W = land.shape[0]
    comp = np.full((W, W), -1, dtype=np.int32)
    cid = 0
    for r in range(W):
        for c in range(W):
            if land[r, c] and comp[r, c] < 0:
                comp[r, c] = cid
                q = deque([(r, c)])
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < W and 0 <= nb < W and land[na, nb] and comp[na, nb] < 0:
                            comp[na, nb] = cid
                            q.append((na, nb))
                cid += 1
    return comp


def _scramble(W: int, rng: np.random.Generator, noise: float = 0.03) -> np.ndarray:
    ys, xs = np.mgrid[0:W, 0:W] / max(W - 1, 1)
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1) - 0.5     # (N,2) centred
    theta = rng.uniform(0, 2 * np.pi)
    shear = rng.uniform(-0.3, 0.3)
    R = np.array([[np.cos(theta), -np.sin(theta) + shear],
                  [np.sin(theta) + shear, np.cos(theta)]])
    uv = xy @ R.T + rng.normal(0, noise, xy.shape)
    return uv                                                  # (N,2)


def make_instance(W: int, rng: np.random.Generator, want_label: int | None = None,
                  max_tries: int = 200) -> LGRCInstance | None:
    for _ in range(max_tries):
        f = _smooth_field(W, rng)
        land = f > np.median(f)
        comp = _components(land)
        land_idx = np.flatnonzero(land.ravel())
        if land_idx.size < 4 or comp.max() < 1:               # need ≥2 components
            continue
        s, t = rng.choice(land_idx, size=2, replace=False)
        same = int(comp.ravel()[s] == comp.ravel()[t])
        if want_label is not None and same != want_label:
            continue
        uv = _scramble(W, rng)
        rc = np.stack(np.divmod(np.arange(W * W), W), axis=1)  # (N,2) true (row,col)
        feats = np.zeros((W * W, N_FEAT), dtype=np.float32)
        feats[:, 0:2] = uv
        feats[:, 2] = land.ravel().astype(np.float32)
        feats[s, 3] = 1.0
        feats[t, 4] = 1.0
        # shuffle set order so row index carries no positional signal
        perm = rng.permutation(W * W)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        return LGRCInstance(feats=feats[perm], label=same, W=W,
                            s_idx=int(inv[s]), t_idx=int(inv[t]),
                            true_rc=rc[perm].astype(np.int64))
    return None


def build(n: int, W: int, seed: int) -> list[LGRCInstance]:
    """n instances, balanced 50/50 by construction."""
    rng = np.random.default_rng(seed)
    out: list[LGRCInstance] = []
    half = n // 2
    for want in (1, 0):
        got = 0
        while got < half:
            inst = make_instance(W, rng, want_label=want)
            if inst is not None:
                out.append(inst)
                got += 1
    rng.shuffle(out)
    return out
