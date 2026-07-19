"""A.0.8b (second-review round 2): the trained curriculum-UT on BOUNDED-DEPTH
tasks (F4b, F5), where 16 iterations suffice at every evaluated width.

The reviewer's point: on F1 (depth ~ W) the UT's fixed 16 iterations make
large-W transfer failure partly architectural. F4b (~K=6 steps) and F5 (T=2
steps) remove that confound: if the trained global still fails to transfer
here, the cause is isolated to the absence of locality, not depth budget.

Recipe identical to a08_curriculum.py except the task family. Pre-registered
in runs/phase12_2d_compute_medium/STAGE_A08B_PREREG.md (committed before the
run).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import device, tensorise
from .families import build_f4b, build_f5
from .models import UniversalTransformer

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
CURRICULUM = [([4], 4 if QUICK else 12),
              ([4, 5], 4 if QUICK else 12),
              ([4, 5, 6], 6 if QUICK else 24)]
EVAL_WS = [4, 6, 8, 12] if QUICK else [4, 5, 6, 8, 10, 12, 16, 20]
SEEDS = [0] if QUICK else [0, 1, 2]
HIDDEN, ITERS = 64, 16
N_TRAIN = 1200 if QUICK else 3000
N_TEST = 500 if QUICK else 1000
WALL_P = 0.28
WARMUP_STEPS, LR = 300, 1e-3

FAMILIES = {
    "F4b": lambda n, W, seed: build_f4b(n, W, W, WALL_P, seed),
    "F5": lambda n, W, seed: build_f5(n, W, W, seed),
}


def _pack(instances, dev):
    X, idx, y, _ = tensorise(instances)
    return {"X": torch.from_numpy(X).to(dev), "idx": torch.from_numpy(idx).to(dev),
            "y": torch.from_numpy(y).to(dev)}


def train_curriculum(family: str, seed: int) -> dict:
    build = FAMILIES[family]
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = UniversalTransformer(HIDDEN, iters=ITERS).to(dev)
    all_ws = sorted({w for ws, _ in CURRICULUM for w in ws})
    tr = {W: _pack(build(N_TRAIN, W, 100 + seed * 17 + W), dev) for W in all_ws}
    te = {W: _pack(build(N_TEST, W, 900 + seed * 17 + W), dev) for W in EVAL_WS}
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / WARMUP_STEPS))
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 128
    for ws, epochs in CURRICULUM:
        for _ in range(epochs):
            model.train()
            units = []
            for W in ws:
                n = tr[W]["y"].shape[0]
                perm = torch.randperm(n, device=dev)
                units += [(W, perm[i:i + bs]) for i in range(0, n, bs)]
            for j in torch.randperm(len(units)).tolist():
                W, sl = units[j]
                p = tr[W]
                opt.zero_grad()
                loss = lossf(model(p["X"][sl], p["idx"][sl]), p["y"][sl])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for W in EVAL_WS:
            p = te[W]
            preds = []
            for i in range(0, p["y"].shape[0], 256):
                preds.append((model(p["X"][i:i + 256], p["idx"][i:i + 256]) > 0).float())
            out[W] = float((torch.cat(preds) == p["y"]).float().mean())
    return out


def verdict(mean: dict) -> dict:
    train_range = [w for w in (4, 5, 6) if w in EVAL_WS]
    trained = float(np.mean([mean[w] for w in train_range]))
    far = [w for w in (16, 20) if w in EVAL_WS]
    far_acc = float(np.mean([mean[w] for w in far])) if far else float("nan")
    trains = trained >= 0.80
    if not trains:
        transfer = "n/a (training failure)"
    elif far_acc <= 0.60:
        transfer = "no transfer"
    elif mean.get(20, mean[max(EVAL_WS)]) >= 0.80:
        transfer = "transfers"
    else:
        transfer = "partial"
    return {"trains_mean_456": trained, "trains": bool(trains),
            "far_mean_16_20": far_acc, "transfer": transfer}


def main() -> None:
    t0 = time.time()
    print(f"A.0.8b curriculum-UT on bounded-depth families  eval={EVAL_WS} "
          f"seeds={SEEDS} hidden={HIDDEN} iters={ITERS}")
    results = {}
    for family in FAMILIES:
        per_seed = [train_curriculum(family, s) for s in SEEDS]
        mean = {W: float(np.mean([r[W] for r in per_seed])) for W in EVAL_WS}
        v = verdict(mean)
        print(f"  [{family}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in EVAL_WS))
        print(f"  [{family}] trains={v['trains_mean_456']:.3f} "
              f"far(16,20)={v['far_mean_16_20']:.3f} -> {v['transfer']}")
        results[family] = {
            "per_seed": [{str(W): r[W] for W in EVAL_WS} for r in per_seed],
            "mean": {str(W): mean[W] for W in EVAL_WS}, **v}
    out = {"config": {"curriculum": CURRICULUM, "eval_ws": EVAL_WS, "seeds": SEEDS,
                      "hidden": HIDDEN, "iters": ITERS, "warmup": WARMUP_STEPS,
                      "lr": LR, "wall_p": WALL_P, "quick": QUICK},
           "families": results,
           "runtime_sec": round(time.time() - t0, 1)}
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "a08b_global_bounded.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"runtime={out['runtime_sec']}s\nwrote {rep / 'a08b_global_bounded.json'}")


if __name__ == "__main__":
    main()
