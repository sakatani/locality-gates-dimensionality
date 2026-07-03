"""Phase 12 — Stage B.1: length-generalisation (PLAN; thresholds in
``STAGE_B1_PREREG.md``). Train on small W ∈ {4,5,6}, test on unseen larger W.

A 2D-local propagation rule is scale-equivariant (transfers across W, bounded by
depth); the row-major 1D serialisation is W-dependent (miscalibrated at test W).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .calibrate import REPORT_DIR, device, tensorise
from .families import build_f1
from .models import (GNNAdj, Local1D, Local1DCurve, Local2D,
                     build_neighbour_index)
from .substrate import gilbert_order

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TRAIN_WS = [4, 5, 6]
TEST_WS = [7, 8, 9, 10] if QUICK else [7, 8, 9, 10, 12]
SEEDS = [0, 1] if QUICK else [0, 1, 2]
DEPTH, HIDDEN = 16, 32
N_TRAIN = 1500 if QUICK else 3000
N_TEST = 600 if QUICK else 1000
EPOCHS = 10 if QUICK else 22
WALL_P = 0.28


def _pack(instances, dev):
    X, idx, y, walls = tensorise(instances)
    return {"X": torch.from_numpy(X).to(dev), "idx": torch.from_numpy(idx).to(dev),
            "y": y, "walls": walls}


def _forward(model, arm, pack, dev, W):
    if arm == "gnn":
        nbr, mask = build_neighbour_index(pack["walls"])
        return model(pack["X"], pack["idx"], torch.from_numpy(nbr).to(dev),
                     torch.from_numpy(mask).to(dev))
    if arm == "1d_hilbert":
        perm = gilbert_order(W, W)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        return model(pack["X"], pack["idx"], torch.from_numpy(perm).to(dev),
                     torch.from_numpy(inv).to(dev))
    return model(pack["X"], pack["idx"])


def train_lengthgen(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = {"2d": Local2D, "1d": Local1D, "1d_hilbert": Local1DCurve,
             "gnn": GNNAdj}[arm](HIDDEN, DEPTH).to(dev)

    train_packs = {W: _pack(build_f1(N_TRAIN, W, W, WALL_P, 100 + seed * 7 + W), dev)
                   for W in TRAIN_WS}
    test_packs = {W: _pack(build_f1(N_TEST, W, W, WALL_P, 700 + seed * 7 + W), dev)
                  for W in TEST_WS}

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 256
    # Work units: (W, batch-slice) across all train sizes, reshuffled each epoch.
    for _ in range(EPOCHS):
        model.train()
        units = []
        for W in TRAIN_WS:
            n = train_packs[W]["y"].shape[0]
            perm = torch.randperm(n, device=dev)
            units += [(W, perm[s:s + bs]) for s in range(0, n, bs)]
        for j in torch.randperm(len(units)).tolist():
            W, bidx = units[j]
            p = train_packs[W]
            sub = {"X": p["X"][bidx], "idx": p["idx"][bidx], "walls": p["walls"][bidx.cpu().numpy()]}
            opt.zero_grad()
            logit = _forward(model, arm, sub, dev, W)
            yb = torch.from_numpy(p["y"][bidx.cpu().numpy()]).to(dev)
            loss = lossf(logit, yb)
            loss.backward()
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for W in TEST_WS:
            p = test_packs[W]
            pred = (_forward(model, arm, p, dev, W) > 0).float().cpu().numpy()
            out[W] = float((pred == p["y"]).mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"Stage B.1 length-gen  train={TRAIN_WS} test={TEST_WS} seeds={SEEDS}")
    arms = ["2d", "1d", "1d_hilbert", "gnn"]
    results = {}
    for arm in arms:
        per_seed = [train_lengthgen(arm, s) for s in SEEDS]
        mean = {W: float(np.mean([r[W] for r in per_seed])) for W in TEST_WS}
        results[arm] = mean
        print(f"  [{arm}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in TEST_WS))

    head_ws = [w for w in TEST_WS if w <= 10]
    gaps = [results["2d"][W] - results["1d"][W] for W in head_ws]
    mean_gap = float(np.mean(gaps))
    all_2d_ge_1d = all(results["2d"][W] >= results["1d"][W] for W in head_ws)
    gnn_control = results["gnn"].get(8, 0.0) >= 0.70
    verdict = {
        "mean_gap_2d_minus_1d": mean_gap,
        "all_2d_ge_1d": bool(all_2d_ge_1d),
        "gnn_control_generalises": bool(gnn_control),
        "PASS": bool(mean_gap >= 0.10 and all_2d_ge_1d and gnn_control),
    }
    out = {"config": {"train_ws": TRAIN_WS, "test_ws": TEST_WS, "seeds": SEEDS,
                      "depth": DEPTH, "hidden": HIDDEN, "epochs": EPOCHS},
           "results": results, "verdict": verdict,
           "runtime_sec": round(time.time() - t0, 1)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "lengthgen.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: PASS={verdict['PASS']}  (mean gap(2D-1D) over W≤10={mean_gap:+.3f} "
          f"need≥0.10; 2D≥1D everywhere={all_2d_ge_1d}; GNN control@8={results['gnn'].get(8,0):.2f}≥0.70="
          f"{gnn_control})  runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'lengthgen.json'}")


if __name__ == "__main__":
    main()
