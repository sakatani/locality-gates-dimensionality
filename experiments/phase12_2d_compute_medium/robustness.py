"""Paper §5.2 robustness checks (reviewer-response experiments).

Part B  — placement-neutral F1 (S, T uniform random; no top/bottom forcing):
          does the 2D−1D length-generalization gap survive without the
          adversarial placement?  Arms 2d/1d, 5 seeds, train W∈{4,5,6},
          test unseen W∈{8,10,12,16,20}; one-sided paired t on the mean gap.
Part C1 — stabilized recurrent GNN (RecGRU-E-lite per Grötschla et al. 2022:
          input-skip + edge MLP + GRU + L2 state reg, variable train rounds,
          3W rounds at inference) on the ORIGINAL F1: does a stabilized
          recurrent GNN avoid the collapse of the plain fixed-depth MPNN?
          3 seeds (diagnostic arm).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

from .calibrate import device, tensorise
from .families import build_f1, build_f1_rand
from .models import Local1D, Local2D, RecGNN, build_neighbour_index

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TRAIN_WS = [4, 5, 6]
TEST_WS = [8, 12, 16] if QUICK else [8, 10, 12, 16, 20]
SEEDS_B = [0, 1] if QUICK else [0, 1, 2, 3, 4]
SEEDS_C = [0] if QUICK else [0, 1, 2]
HIDDEN, DEPTH = 32, 16
N_TRAIN = 1500 if QUICK else 3000
N_TEST = 600 if QUICK else 1000
EPOCHS = 10 if QUICK else 22
# The recurrent-GNN baseline gets a LARGER budget (fair-shot principle, cf.
# the parameter-parity Universal Transformer in A.0.7): recurrent GNNs train
# slowly (Grötschla et al. used 100 epochs), so C1 runs 40 epochs.
EPOCHS_C = 12 if QUICK else 40
WALL_P = 0.28
L2_COEF = 1e-2


def _pack(instances, dev, with_nbr: bool = False):
    X, idx, y, walls = tensorise(instances)
    p = {"X": torch.from_numpy(X).to(dev), "idx": torch.from_numpy(idx).to(dev),
         "y": torch.from_numpy(y).to(dev)}
    if with_nbr:
        nbr, mask = build_neighbour_index(walls)
        p["nbr"] = torch.from_numpy(nbr).to(dev)
        p["mask"] = torch.from_numpy(mask).to(dev)
    return p


# ---------------------------------------------------------------------------
# Part B — placement-neutral F1, local arms
# ---------------------------------------------------------------------------

def train_local(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = {"2d": Local2D, "1d": Local1D}[arm](HIDDEN, DEPTH).to(dev)
    tr = {W: _pack(build_f1_rand(N_TRAIN, W, W, WALL_P, 100 + seed * 13 + W), dev)
          for W in TRAIN_WS}
    te = {W: _pack(build_f1_rand(N_TEST, W, W, WALL_P, 900 + seed * 13 + W), dev)
          for W in TEST_WS}
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 256
    for _ in range(EPOCHS):
        model.train()
        units = []
        for W in TRAIN_WS:
            n = tr[W]["y"].shape[0]
            perm = torch.randperm(n, device=dev)
            units += [(W, perm[i:i + bs]) for i in range(0, n, bs)]
        for j in torch.randperm(len(units)).tolist():
            W, sl = units[j]
            p = tr[W]
            opt.zero_grad()
            loss = lossf(model(p["X"][sl], p["idx"][sl]), p["y"][sl])
            loss.backward()
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for W in TEST_WS:
            p = te[W]
            pred = (model(p["X"], p["idx"]) > 0).float()
            out[W] = float((pred == p["y"]).float().mean())
    return out


# ---------------------------------------------------------------------------
# Part C1 — stabilized recurrent GNN on original F1
# ---------------------------------------------------------------------------

def train_recgnn(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = RecGNN(HIDDEN).to(dev)
    tr = {W: _pack(build_f1(N_TRAIN, W, W, WALL_P, 100 + seed * 13 + W), dev,
                   with_nbr=True) for W in TRAIN_WS}
    te = {W: _pack(build_f1(N_TEST, W, W, WALL_P, 900 + seed * 13 + W), dev,
                   with_nbr=True) for W in TEST_WS}
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 256
    for _ in range(EPOCHS_C):
        model.train()
        units = []
        for W in TRAIN_WS:
            n = tr[W]["y"].shape[0]
            perm = torch.randperm(n, device=dev)
            units += [(W, perm[i:i + bs]) for i in range(0, n, bs)]
        for j in torch.randperm(len(units)).tolist():
            W, sl = units[j]
            p = tr[W]
            rounds = int(rng.integers(W, 3 * W + 1))   # variable-rounds recipe
            opt.zero_grad()
            logit, l2 = model(p["X"][sl], p["idx"][sl], p["nbr"][sl],
                              p["mask"][sl], rounds)
            loss = lossf(logit, p["y"][sl]) + L2_COEF * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for W in TEST_WS:
            p = te[W]
            logit, _ = model(p["X"], p["idx"], p["nbr"], p["mask"], 3 * W)
            out[W] = float(((logit > 0).float() == p["y"]).float().mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"robustness  train={TRAIN_WS} test={TEST_WS} seedsB={SEEDS_B} "
          f"seedsC={SEEDS_C} quick={QUICK}")

    print("[B] placement-neutral F1 (random S,T):")
    b: dict = {}
    for arm in ("2d", "1d"):
        per_seed = [train_local(arm, s) for s in SEEDS_B]
        b[arm] = {W: [r[W] for r in per_seed] for W in TEST_WS}
        mean = {W: float(np.mean(v)) for W, v in b[arm].items()}
        print(f"  [{arm}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in TEST_WS))
    gaps = [float(np.mean([b["2d"][W][i] - b["1d"][W][i] for W in TEST_WS]))
            for i in range(len(SEEDS_B))]
    g = np.asarray(gaps)
    tstat, p_two = sps.ttest_1samp(g, 0.0)
    p_one = float(p_two / 2 if tstat > 0 else 1 - p_two / 2)
    print(f"  gap(2D−1D) mean={g.mean():+.3f}±{g.std(ddof=1):.3f} "
          f"t={tstat:.2f} p_one={p_one:.2e}")

    print("[C1] stabilized recurrent GNN (RecGRU-E-lite) on original F1:")
    per_seed = [train_recgnn(s) for s in SEEDS_C]
    c1 = {W: [r[W] for r in per_seed] for W in TEST_WS}
    mean_c1 = {W: float(np.mean(v)) for W, v in c1.items()}
    print("  [recgnn] " + " ".join(f"{W}:{mean_c1[W]:.2f}" for W in TEST_WS))

    out = {"config": {"train_ws": TRAIN_WS, "test_ws": TEST_WS,
                      "seeds_b": SEEDS_B, "seeds_c": SEEDS_C,
                      "hidden": HIDDEN, "depth": DEPTH, "epochs": EPOCHS,
                      "epochs_c": EPOCHS_C, "l2_coef": L2_COEF, "quick": QUICK},
           "f1rand": {arm: {str(W): v for W, v in d.items()} for arm, d in b.items()},
           "f1rand_gap": {"per_seed": [round(x, 4) for x in gaps],
                          "mean": float(g.mean()), "sd": float(g.std(ddof=1)),
                          "t": float(tstat), "p_one_sided": p_one},
           "recgnn_f1": {str(W): v for W, v in c1.items()},
           "runtime_sec": round(time.time() - t0, 1)}
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"runtime={out['runtime_sec']}s\nwrote {rep / 'robustness.json'}")


if __name__ == "__main__":
    main()
