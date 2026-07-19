"""E1 (second-review response): a SERIOUS attempt at a trainable
iterative-global baseline on F1 — curriculum over W + lr warmup + generous
capacity — so the §7.1 boundary rests on a trained model, not a training
failure. Third and best-equipped attempt (after looped @A.0.6 and param-parity
UT @A.0.7).

Recipe: UniversalTransformer (weight-shared, pre-LN, per-step embeddings),
hidden 64, 16 iterations; curriculum W=4 → {4,5} → {4,5,6}; linear lr warmup
(300 steps to 1e-3) then constant; grad clip 1.0; 3 seeds.

Read-out: does it TRAIN (mean acc ≥ 0.80 at W ∈ {4,5,6})? If yes, how does it
size-generalize to unseen W (8..20)? Either outcome informs §7.1: a trained
global that solves all widths CONFIRMS "given global attention, layout is
unnecessary"; a trained global that fails to transfer would strengthen the 2D
case beyond current claims; a third training failure is reported as such.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import device, tensorise
from .families import build_f1
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


def _pack(instances, dev):
    X, idx, y, _ = tensorise(instances)
    return {"X": torch.from_numpy(X).to(dev), "idx": torch.from_numpy(idx).to(dev),
            "y": torch.from_numpy(y).to(dev)}


def train_curriculum(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = UniversalTransformer(HIDDEN, iters=ITERS).to(dev)
    all_ws = sorted({w for ws, _ in CURRICULUM for w in ws})
    tr = {W: _pack(build_f1(N_TRAIN, W, W, WALL_P, 100 + seed * 17 + W), dev)
          for W in all_ws}
    te = {W: _pack(build_f1(N_TEST, W, W, WALL_P, 900 + seed * 17 + W), dev)
          for W in EVAL_WS}
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
            for i in range(0, p["y"].shape[0], 256):     # chunk large-W attention
                preds.append((model(p["X"][i:i + 256], p["idx"][i:i + 256]) > 0).float())
            out[W] = float((torch.cat(preds) == p["y"]).float().mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"E1 curriculum-UT  curriculum={CURRICULUM} eval={EVAL_WS} seeds={SEEDS} "
          f"hidden={HIDDEN} iters={ITERS}")
    per_seed = [train_curriculum(s) for s in SEEDS]
    mean = {W: float(np.mean([r[W] for r in per_seed])) for W in EVAL_WS}
    print("  [univ-curr] " + " ".join(f"{W}:{mean[W]:.2f}" for W in EVAL_WS))
    train_range = [w for w in (4, 5, 6) if w in EVAL_WS]
    trained = float(np.mean([mean[w] for w in train_range]))
    print(f"  trains(mean acc @W in {train_range}) = {trained:.3f} (bar 0.80)")
    out = {"config": {"curriculum": CURRICULUM, "eval_ws": EVAL_WS, "seeds": SEEDS,
                      "hidden": HIDDEN, "iters": ITERS, "warmup": WARMUP_STEPS,
                      "lr": LR, "quick": QUICK},
           "per_seed": [{str(W): r[W] for W in EVAL_WS} for r in per_seed],
           "mean": {str(W): mean[W] for W in EVAL_WS},
           "trains_mean_456": trained, "trains": bool(trained >= 0.80),
           "runtime_sec": round(time.time() - t0, 1)}
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "a08_curriculum.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"runtime={out['runtime_sec']}s\nwrote {rep / 'a08_curriculum.json'}")


if __name__ == "__main__":
    main()
