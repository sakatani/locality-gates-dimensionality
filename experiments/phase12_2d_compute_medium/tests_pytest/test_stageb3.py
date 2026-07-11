"""Phase 12 Stage B.3 unit tests — F4 metric + F5 dynamics families."""

from __future__ import annotations

import numpy as np

from experiments.phase12_2d_compute_medium.families import (
    build_f4, build_f5, class_balance, instance_to_channels, readout_cell)
from experiments.phase12_2d_compute_medium.substrate import (
    bfs_distance, ca_step, reachable)


def test_bfs_distance_matches_reachable():
    walls = np.zeros((5, 5), dtype=np.uint8)
    walls[1:4, 2] = 1                       # a wall column with a gap at row 4
    d = bfs_distance(walls, (0, 0))
    reach = reachable(walls, (0, 0))
    assert (d >= 0).sum() == reach.sum()    # same reachable set
    assert d[0, 0] == 0 and d[0, 1] == 1    # adjacency correct


def test_ca_step_is_local_and_deterministic():
    s = np.zeros((3, 3), dtype=np.uint8)
    s[1, 1] = 1                             # single ON cell: total=1 → dies
    assert ca_step(s)[1, 1] == 0
    s2 = np.zeros((3, 3), dtype=np.uint8)
    s2[1, 1] = 1; s2[0, 1] = 1              # center sees self+1 nb = 2 → ON
    assert ca_step(s2)[1, 1] == 1
    assert np.array_equal(ca_step(s2), ca_step(s2))  # deterministic


def test_f4_balanced_and_label_correct():
    data = build_f4(200, 7, 7, wall_p=0.28, seed=0)
    assert abs(class_balance(data) - 0.5) < 0.06
    K = 2 * (7 - 1)
    for inst in data[:40]:
        d = int(bfs_distance(inst.walls, inst.source)[inst.target])
        assert inst.label == int(0 <= d <= K)
        assert readout_cell(inst) == inst.target


def test_f5_balanced_and_label_correct():
    data = build_f5(200, 7, 7, seed=0, steps=2)
    assert abs(class_balance(data) - 0.5) < 0.06
    for inst in data[:40]:
        rolled = inst.values.astype(np.uint8)
        for _ in range(2):
            rolled = ca_step(rolled)
        assert inst.label == int(rolled[inst.target])
        assert readout_cell(inst) == inst.target


def test_f4_f5_tensorise_shape():
    for inst in (build_f4(4, 6, 6, 0.28, 1)[0], build_f5(4, 6, 6, 1)[0]):
        ch = instance_to_channels(inst)
        assert ch.shape == (6, 6, 6)
