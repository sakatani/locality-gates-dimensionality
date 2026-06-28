"""Phase 12 Stage A.0 unit tests — substrate, families, model wiring."""

from __future__ import annotations

import numpy as np
import torch

from experiments.phase12_2d_compute_medium.families import (
    N_CHANNELS, build_f1, build_f3, class_balance, instance_to_channels,
    readout_cell)
from experiments.phase12_2d_compute_medium.models import (
    GNNAdj, GlobalAttn, Local1D, Local2D, build_neighbour_index)
from experiments.phase12_2d_compute_medium.substrate import (
    OPEN, WALL, GridInstance, bandwidth_2d, grid_bandwidth_best_1d,
    grid_edges, ordering_bandwidth, reachable, row_major_bandwidth)


# --- bandwidth decoupling (the PLAN §1 core claim) -------------------------

def test_row_major_bandwidth_equals_W():
    for w in (4, 7, 9, 12):
        assert row_major_bandwidth(w, w) == w


def test_best_1d_bandwidth_is_min_wh():
    assert grid_bandwidth_best_1d(9, 5) == 5
    assert grid_bandwidth_best_1d(5, 9) == 5
    assert grid_bandwidth_best_1d(7, 7) == 7


def test_2d_bandwidth_constant():
    assert bandwidth_2d() == 1


def test_bandwidth_knob_decouples():
    # 2D-bw constant while best-1D-bw grows with W → the knob separates them.
    bw2 = [bandwidth_2d() for _ in (5, 7, 9, 11)]
    bw1 = [grid_bandwidth_best_1d(w, w) for w in (5, 7, 9, 11)]
    assert bw2 == [1, 1, 1, 1]
    assert bw1 == [5, 7, 9, 11]


def test_ordering_bandwidth_basic():
    edges = grid_edges(3, 3)
    assert ordering_bandwidth(edges, np.arange(9)) == 3  # vertical Δ = W = 3


# --- reachability ----------------------------------------------------------

def test_reachable_open_grid():
    walls = np.zeros((4, 4), np.uint8)
    reach = reachable(walls, (0, 0))
    assert reach.all()


def test_reachable_blocked():
    walls = np.zeros((3, 3), np.uint8)
    walls[1, :] = WALL  # full horizontal wall splits top from bottom
    reach = reachable(walls, (0, 0))
    assert reach[0].all()
    assert not reach[2].any()


# --- F1 family validity ----------------------------------------------------

def test_f1_balanced_and_labels_correct():
    data = build_f1(400, 7, 7, wall_p=0.28, seed=0)
    assert len(data) > 0
    bal = class_balance(data)
    assert 0.4 <= bal <= 0.6  # ~50/50
    for inst in data[:50]:
        assert inst.source[0] == 0           # source in top row
        assert inst.target[0] == inst.H - 1  # target in bottom row
        recomputed = int(reachable(inst.walls, inst.source)[inst.target])
        assert recomputed == inst.label


def test_f1_density_matched():
    data = build_f1(400, 7, 7, wall_p=0.28, seed=1)
    pos = [i.walls.sum() for i in data if i.label == 1]
    neg = [i.walls.sum() for i in data if i.label == 0]
    # Exact wall-count pairing → mean density equal by construction.
    assert abs(np.mean(pos) - np.mean(neg)) < 0.5


# --- F3 control validity ---------------------------------------------------

def test_f3_label_is_row_prefix_parity():
    data = build_f3(100, 8, 8, seed=0)
    for inst in data:
        r0, c0 = inst.source
        expected = int(inst.values[r0, : c0 + 1].sum()) % 2
        assert inst.label == expected
        assert readout_cell(inst) == inst.source


def test_f3_deterministic():
    a = build_f3(50, 6, 6, seed=3)
    b = build_f3(50, 6, 6, seed=3)
    assert all(x.label == y.label for x, y in zip(a, b))


# --- tensorisation parity (H1) ---------------------------------------------

def test_channels_shape_and_markers():
    data = build_f1(10, 5, 5, wall_p=0.3, seed=0)
    ch = instance_to_channels(data[0])
    assert ch.shape == (N_CHANNELS, 5, 5)
    assert ch[1][data[0].source] == 1.0  # is_source
    assert ch[2][data[0].target] == 1.0  # is_target


# --- model wiring (tiny forward passes) ------------------------------------

def _tiny_batch():
    data = build_f1(8, 5, 5, wall_p=0.3, seed=0)
    X = torch.from_numpy(np.stack([instance_to_channels(i) for i in data]).astype(np.float32))
    idx = torch.tensor([r * i.W + c for i in data for (r, c) in [readout_cell(i)]])
    walls = np.stack([i.walls for i in data]).astype(np.int64)
    return X, idx, walls


def test_local_arms_forward():
    X, idx, _ = _tiny_batch()
    for model in (Local2D(16, 4), Local1D(16, 4)):
        out = model(X, idx)
        assert out.shape == (8,)


def test_global_arm_forward():
    X, idx, _ = _tiny_batch()
    out = GlobalAttn(16, layers=2)(X, idx)
    assert out.shape == (8,)


def test_gnn_arm_forward():
    X, idx, walls = _tiny_batch()
    nbr, mask = build_neighbour_index(walls)
    out = GNNAdj(16, 4)(X, idx, torch.from_numpy(nbr), torch.from_numpy(mask))
    assert out.shape == (8,)


def test_matched_param_count_2d_vs_1d():
    # H2: 3×3 conv2d and kernel-9 conv1d have identical param counts.
    p2 = sum(p.numel() for p in Local2D(32, 12).parameters())
    p1 = sum(p.numel() for p in Local1D(32, 12).parameters())
    assert p2 == p1
