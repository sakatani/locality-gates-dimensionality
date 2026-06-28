"""Phase 12 Stage A.0.6 unit tests — space-filling order + hardening arms."""

from __future__ import annotations

import numpy as np
import torch

from experiments.phase12_2d_compute_medium.families import (
    build_f1, instance_to_channels, readout_cell)
from experiments.phase12_2d_compute_medium.models import (
    GlobalLooped, Local1DCurve)
from experiments.phase12_2d_compute_medium.substrate import gilbert_order


def test_gilbert_is_permutation():
    for W, H in [(4, 4), (6, 6), (8, 8), (6, 4), (5, 7)]:
        order = gilbert_order(W, H)
        assert sorted(order.tolist()) == list(range(W * H))


def test_gilbert_steps_are_4adjacent():
    # Every consecutive pair in a space-filling curve is a 4-neighbour.
    for W, H in [(4, 4), (8, 8), (6, 4)]:
        order = gilbert_order(W, H)
        rc = [(i // W, i % W) for i in order]
        for (r0, c0), (r1, c1) in zip(rc, rc[1:]):
            assert abs(r0 - r1) + abs(c0 - c1) == 1


def _tiny_batch(W=6):
    data = build_f1(8, W, W, wall_p=0.28, seed=0)
    X = torch.from_numpy(
        np.stack([instance_to_channels(i) for i in data]).astype(np.float32))
    idx = torch.tensor([r * i.W + c for i in data for (r, c) in [readout_cell(i)]])
    return X, idx, W


def test_curve_arm_forward():
    X, idx, W = _tiny_batch()
    perm = gilbert_order(W, W)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    out = Local1DCurve(16, 4)(X, idx, torch.from_numpy(perm), torch.from_numpy(inv))
    assert out.shape == (8,)


def test_looped_global_forward():
    X, idx, _ = _tiny_batch()
    out = GlobalLooped(16, iters=4)(X, idx)
    assert out.shape == (8,)


def test_curve_arm_matches_1d_param_count():
    from experiments.phase12_2d_compute_medium.models import Local1D
    p_curve = sum(p.numel() for p in Local1DCurve(32, 12).parameters())
    p_1d = sum(p.numel() for p in Local1D(32, 12).parameters())
    assert p_curve == p_1d  # same operator, only the order differs
