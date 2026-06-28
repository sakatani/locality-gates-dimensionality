"""Phase 12 — task families (PLAN §4).

F1  2D multi-hop propagation (reachability). Low 2D-bandwidth, high 1D-bandwidth.
    Knob W. The S→T path must traverse the grid *vertically* (S in the top row,
    T in the bottom row) — the high-1D-bandwidth direction — so a 1D-local
    (row-major) operator is handicapped by the factor W.

F3  1D-intrinsic control (refutation). Prefix parity along the source cell's
    *row*: the label depends only on one horizontal axis, so a 2D layout gives
    no advantage over the row-major 1D layout. Here 2D-local must NOT beat
    1D-local — if it does, the F1 win was generic capacity, not 2D bandwidth.

Validity (PLAN §4): F1 is class-balanced 50/50 AND wall-count-matched between
classes, so the reachable/unreachable answer cannot be read off aggregate wall
density — it genuinely requires propagating connectivity.
"""

from __future__ import annotations

import numpy as np

from .substrate import OPEN, WALL, GridInstance, reachable


# ---------------------------------------------------------------------------
# F1 — reachability
# ---------------------------------------------------------------------------

def _random_grid(W: int, H: int, wall_p: float, rng: np.random.Generator):
    """Random walls; S in the top row, T in the bottom row (forces vertical path)."""
    walls = (rng.random((H, W)) < wall_p).astype(np.uint8)
    sc = int(rng.integers(W))
    tc = int(rng.integers(W))
    source = (0, sc)
    target = (H - 1, tc)
    walls[source] = OPEN
    walls[target] = OPEN
    return walls, source, target


def _make_f1_instance(W, H, wall_p, rng) -> tuple[GridInstance, int]:
    walls, source, target = _random_grid(W, H, wall_p, rng)
    reach = reachable(walls, source)
    label = int(reach[target])
    inst = GridInstance(walls=walls, source=source, target=target,
                        label=label, family="F1")
    return inst, int(walls.sum())


def build_f1(n: int, W: int, H: int, wall_p: float, seed: int,
             max_tries: int = 400_000) -> list[GridInstance]:
    """n instances, 50/50 reachable, **exact** wall-count matched between classes.

    Each reachable (label 1) instance is paired with an unreachable (label 0)
    instance of the *identical* wall count, so the matched multiset of wall
    counts is the same for both classes → mean density is equal by construction
    and the reachable/unreachable answer cannot be read off aggregate density
    (PLAN §4 validity). Oversamples candidates so the yield stays near n.
    """
    rng = np.random.default_rng(seed)
    pos: dict[int, list[GridInstance]] = {}
    neg: dict[int, list[GridInstance]] = {}
    half = n // 2
    tries = 0
    while (sum(len(v) for v in pos.values()) < n
           or sum(len(v) for v in neg.values()) < n) and tries < max_tries:
        tries += 1
        inst, wc = _make_f1_instance(W, H, wall_p, rng)
        (pos if inst.label == 1 else neg).setdefault(wc, []).append(inst)
    out: list[GridInstance] = []
    for wc, plist in pos.items():
        nlist = neg.get(wc, [])
        for p, q in zip(plist, nlist):
            if len(out) >= 2 * half:
                break
            out.append(p)
            out.append(q)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# F3 — 1D-intrinsic control (row prefix parity)
# ---------------------------------------------------------------------------

def build_f3(n: int, W: int, H: int, seed: int) -> list[GridInstance]:
    """Each cell carries a random bit. Source (r0, c0) random. Label = parity
    of the bits in row r0, columns 0..c0 inclusive. Vertical axis is irrelevant.
    """
    rng = np.random.default_rng(seed)
    out: list[GridInstance] = []
    for _ in range(n):
        values = (rng.random((H, W)) < 0.5).astype(np.float32)
        r0 = int(rng.integers(H))
        c0 = int(rng.integers(W))
        label = int(values[r0, : c0 + 1].sum()) % 2
        walls = np.zeros((H, W), dtype=np.uint8)  # no walls in F3
        out.append(GridInstance(walls=walls, source=(r0, c0), target=(r0, c0),
                                label=label, family="F3", values=values))
    return out


# ---------------------------------------------------------------------------
# Tensorisation (PLAN H1 — identical info for every arm)
# ---------------------------------------------------------------------------

def instance_to_channels(inst: GridInstance) -> np.ndarray:
    """(C, H, W) feature map shared by all arms.

    Channels: [is_wall, is_source, is_target, value, x_norm, y_norm].
    Positions (x_norm, y_norm) are explicit so the *sequence* and *graph* arms
    have the same information the *grid* arm gets implicitly (H1 parity).
    """
    H, W = inst.H, inst.W
    is_wall = (inst.walls == WALL).astype(np.float32)
    is_src = np.zeros((H, W), np.float32)
    is_tgt = np.zeros((H, W), np.float32)
    is_src[inst.source] = 1.0
    is_tgt[inst.target] = 1.0
    value = inst.values if inst.values is not None else np.zeros((H, W), np.float32)
    xs = np.broadcast_to(np.linspace(0, 1, W, dtype=np.float32)[None, :], (H, W))
    ys = np.broadcast_to(np.linspace(0, 1, H, dtype=np.float32)[:, None], (H, W))
    return np.stack([is_wall, is_src, is_tgt, value.astype(np.float32), xs, ys], axis=0)


N_CHANNELS = 6


def readout_cell(inst: GridInstance) -> tuple[int, int]:
    """Which cell's final embedding carries the answer (target for F1, source for F3)."""
    return inst.target if inst.family == "F1" else inst.source


def class_balance(instances: list[GridInstance]) -> float:
    labels = np.array([i.label for i in instances])
    return float(labels.mean()) if len(labels) else 0.0
