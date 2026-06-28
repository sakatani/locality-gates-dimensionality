"""Phase 12 — layout-as-schedule (LaS) substrate (PLAN §1, §4).

A problem instance is a 2D grid of *cells*. Spatial adjacency (4-neighbour)
is the one-step interaction; the answer is the fixed point of iterated local
updates (reachability, a CA rollout, a row scan). The substrate is **lossless
and identical across arms** (PLAN H1): every model arm consumes the same cells,
the same positions, the same features — they differ only in the *locality
structure of their operator* (2D-local / 1D-local / global / GNN).

Bandwidth (PLAN §1 governing quantity). For the W×H grid graph with 4-adjacency:

- **2D bandwidth = 1** by construction (neighbours are lattice-adjacent).
- **best 1D bandwidth = min(W, H)** (the grid-graph bandwidth; Chvátalová 1975).
  Row-major order realises bandwidth = W (vertical neighbours are W apart).

So the knob W *decouples* the two: 2D-bw stays 1, best-1D-bw grows as W. That
gap is the object Phase 12 tests — see ``ordering_bandwidth`` /
``grid_bandwidth_best_1d`` and the calibration check that it is real.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

WALL = 1
OPEN = 0


@dataclass(frozen=True)
class GridInstance:
    """One problem instance on a W×H grid.

    ``walls`` is (H, W) uint8 (1 = wall, 0 = open). ``values`` is an optional
    (H, W) float field (used by F3). ``source`` / ``target`` are (row, col).
    ``label`` is the binary target. ``family`` tags the generator.
    """

    walls: np.ndarray  # (H, W) uint8
    source: tuple[int, int]
    target: tuple[int, int]
    label: int
    family: str
    values: np.ndarray | None = None  # (H, W) float32, optional

    @property
    def H(self) -> int:
        return int(self.walls.shape[0])

    @property
    def W(self) -> int:
        return int(self.walls.shape[1])


# ---------------------------------------------------------------------------
# Adjacency + reachability
# ---------------------------------------------------------------------------

def neighbours4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r < H - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < W - 1:
        yield r, c + 1


def reachable(walls: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """4-connected flood fill over OPEN cells from ``source`` → (H, W) bool."""
    H, W = walls.shape
    seen = np.zeros((H, W), dtype=bool)
    sr, sc = source
    if walls[sr, sc] == WALL:
        return seen
    seen[sr, sc] = True
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbours4(r, c, H, W):
            if not seen[nr, nc] and walls[nr, nc] == OPEN:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def grid_edges(H: int, W: int) -> list[tuple[int, int]]:
    """All 4-neighbour cell-index pairs (row-major index), walls ignored."""
    edges: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            u = r * W + c
            if c < W - 1:
                edges.append((u, u + 1))      # horizontal: |Δidx| = 1
            if r < H - 1:
                edges.append((u, u + W))      # vertical:   |Δidx| = W
    return edges


def open_edges(walls: np.ndarray) -> list[tuple[int, int]]:
    """4-neighbour pairs where BOTH endpoints are open (the relational graph)."""
    H, W = walls.shape
    edges: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if walls[r, c] == WALL:
                continue
            u = r * W + c
            if c < W - 1 and walls[r, c + 1] == OPEN:
                edges.append((u, u + 1))
            if r < H - 1 and walls[r + 1, c] == OPEN:
                edges.append((u, u + W))
    return edges


# ---------------------------------------------------------------------------
# Bandwidth analytics (PLAN §1, §7 Stage A.0 "verify the knob decouples")
# ---------------------------------------------------------------------------

def ordering_bandwidth(edges: list[tuple[int, int]], order: np.ndarray) -> int:
    """max over edges of |order[u] - order[v]| for a given vertex ordering."""
    if not edges:
        return 0
    e = np.asarray(edges)
    return int(np.abs(order[e[:, 0]] - order[e[:, 1]]).max())


def row_major_bandwidth(H: int, W: int) -> int:
    """Bandwidth of the full grid graph under the row-major order ( = W )."""
    order = np.arange(H * W)
    return ordering_bandwidth(grid_edges(H, W), order)


def grid_bandwidth_best_1d(H: int, W: int) -> int:
    """Best achievable 1D bandwidth of the P_W × P_H grid graph = min(W, H).

    Chvátalová (1975): the bandwidth of the Cartesian product of two paths
    P_m × P_n is min(m, n). This is the *floor* over all 1D linearisations:
    no serialisation keeps every 4-neighbour closer than min(W, H).
    """
    return min(W, H)


def bandwidth_2d() -> int:
    """2D lattice embedding bandwidth = 1 (neighbours are lattice-adjacent)."""
    return 1
