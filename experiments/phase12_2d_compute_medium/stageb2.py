"""Phase 12 — Stage B.2: adaptive depth (PLAN; thresholds in
``STAGE_B2_PREREG.md``). Weight-shared local update iterated T=2W steps vs a
fixed-16-layer 2D baseline, on unseen large W — does adaptive depth break the
depth bound that capped B.1?
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .calibrate import REPORT_DIR, device, tensorise
from .families import build_f1
from .models import ConvCA1D, ConvCA2D, Local2D

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TRAIN_WS = [4, 5, 6]
TEST_WS = [8, 12, 16] if QUICK else [8, 10, 12, 16, 20]
SEEDS = [0, 1] if QUICK else [0, 1, 2]
HIDDEN, FIXED_DEPTH = 32, 16
N_TRAIN = 1500 if QUICK else 3000
N_TEST = 600 if QUICK else 1000
EPOCHS = 10 if QUICK else 22
WALL_P = 0.28


def _test_iters(W: int) -> int:
    return 3 * W                                   # generous propagation headroom


def _train_iters(W: int, rng: np.random.Generator) -> int:
    # Canonical neural-CA recipe: variable step count per batch → the readout
    # becomes a stable attractor robust to running MORE steps at test.
    return int(rng.integers(W, 3 * W + 1))


def _pack(instances, dev):
    X, idx, y, _ = tensorise(instances)
    return {"X": torch.from_numpy(X).to(dev), "idx": torch.from_numpy(idx).to(dev),
            "y": y}


def _forward(model, arm, pack, iters):
    if arm == "fixed_2d":
        return model(pack["X"], pack["idx"])
    return model(pack["X"], pack["idx"], iters)    # ca_2d / ca_1d


def train_adaptive(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = {"ca_2d": lambda: ConvCA2D(HIDDEN),
             "ca_1d": lambda: ConvCA1D(HIDDEN),
             "fixed_2d": lambda: Local2D(HIDDEN, FIXED_DEPTH)}[arm]().to(dev)

    train_packs = {W: _pack(build_f1(N_TRAIN, W, W, WALL_P, 200 + seed * 7 + W), dev)
                   for W in TRAIN_WS}
    test_packs = {W: _pack(build_f1(N_TEST, W, W, WALL_P, 800 + seed * 7 + W), dev)
                  for W in TEST_WS}

    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 256
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
            sub = {"X": p["X"][bidx], "idx": p["idx"][bidx]}
            opt.zero_grad()
            logit = _forward(model, arm, sub, _train_iters(W, rng))
            yb = torch.from_numpy(p["y"][bidx.cpu().numpy()]).to(dev)
            loss = lossf(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for W in TEST_WS:
            p = test_packs[W]
            pred = (_forward(model, arm, p, _test_iters(W)) > 0).float().cpu().numpy()
            out[W] = float((pred == p["y"]).mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"Stage B.2 adaptive-depth  train={TRAIN_WS} test={TEST_WS} iters=2W seeds={SEEDS}")
    results = {}
    for arm in ["ca_2d", "fixed_2d", "ca_1d"]:
        per_seed = [train_adaptive(arm, s) for s in SEEDS]
        mean = {W: float(np.mean([r[W] for r in per_seed])) for W in TEST_WS}
        results[arm] = mean
        print(f"  [{arm}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in TEST_WS))

    ca16 = results["ca_2d"].get(16, 0.0)
    fixed16 = results["fixed_2d"].get(16, 0.0)
    ca1d16 = results["ca_1d"].get(16, 0.0)
    verdict = {
        "ca_2d_at_16": ca16,
        "ca_minus_fixed_at_16": ca16 - fixed16,
        "ca_minus_ca1d_at_16": ca16 - ca1d16,
        "holds_at_16": bool(ca16 >= 0.70),
        "beats_fixed": bool((ca16 - fixed16) >= 0.15),
        "beats_1d": bool((ca16 - ca1d16) >= 0.20),
        "PASS": bool(ca16 >= 0.70 and (ca16 - fixed16) >= 0.15 and (ca16 - ca1d16) >= 0.20),
    }
    out = {"config": {"train_ws": TRAIN_WS, "test_ws": TEST_WS, "seeds": SEEDS,
                      "hidden": HIDDEN, "fixed_depth": FIXED_DEPTH, "epochs": EPOCHS,
                      "iters_rule": "train~U[W,3W] test=3W"},
           "results": results, "verdict": verdict,
           "runtime_sec": round(time.time() - t0, 1)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "stageb2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: PASS={verdict['PASS']}  (ca_2d@16={ca16:.2f}≥0.70={verdict['holds_at_16']}; "
          f"ca−fixed@16={ca16-fixed16:+.2f}≥0.15={verdict['beats_fixed']}; "
          f"ca−ca1d@16={ca16-ca1d16:+.2f}≥0.20={verdict['beats_1d']})  runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'stageb2.json'}")


if __name__ == "__main__":
    main()
