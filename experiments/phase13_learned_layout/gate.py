"""Phase 13 — Stage C.0 cheap gate (PLAN; thresholds in PLAN §5, fixed before run).

Train each arm on small W ∈ {6,8}; test size-generalization on unseen W ∈
{12,16,20}. PASS iff layout-2D (1) learns (≥0.75 @ W=8), (2) beats max(set-tf,
kNN-GNN) by ≥0.10 at the largest test W, (3) beats random-layout by ≥0.10 there.
NULL → pivot to Plan B.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .models import KNNGNN, LayoutCA, SetTransformer
from .task import build

QUICK = os.environ.get("PHASE13_QUICK", "0") == "1"
TRAIN_WS = [6, 8]
EVAL_WS = [8, 12, 16] if QUICK else [8, 12, 16, 20]
SEEDS = [0, 1] if QUICK else [0, 1, 2]
HIDDEN = 32
N_TRAIN = 800 if QUICK else 1500
N_TEST = 400 if QUICK else 800
EPOCHS = 8 if QUICK else 20
ARMS = ["layout", "random", "settf", "knn"]


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _pack(instances, dev, seed):
    rng = np.random.default_rng(seed + instances[0].W)
    X = torch.tensor(np.stack([d.feats for d in instances]), device=dev)
    s = torch.tensor([d.s_idx for d in instances], device=dev)
    t = torch.tensor([d.t_idx for d in instances], device=dev)
    y = torch.tensor([float(d.label) for d in instances], device=dev)
    W = instances[0].W
    rp = torch.tensor(np.stack([rng.permutation(W * W) for _ in instances]), device=dev)
    return {"X": X, "s": s, "t": t, "y": y, "W": W, "rp": rp}


def _forward(model, arm, p, sl=None):
    X, s, t, W = p["X"], p["s"], p["t"], p["W"]
    if sl is not None:
        X, s, t = X[sl], s[sl], t[sl]
    if arm == "layout":
        return model(X, s, t, W)
    if arm == "random":
        rp = p["rp"] if sl is None else p["rp"][sl]
        return model(X, s, t, W, learned=False, rand_perm=rp)
    return model(X, s, t, W)


def train_arm(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    model = {"layout": LayoutCA, "random": LayoutCA,
             "settf": SetTransformer, "knn": KNNGNN}[arm](HIDDEN).to(dev)
    train_packs = {W: _pack(build(N_TRAIN, W, 100 + seed * 9 + W), dev, seed)
                   for W in TRAIN_WS}
    test_packs = {W: _pack(build(N_TEST, W, 700 + seed * 9 + W), dev, seed)
                  for W in EVAL_WS}
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 32
    for _ in range(EPOCHS):
        model.train()
        units = []
        for W in TRAIN_WS:
            n = train_packs[W]["y"].shape[0]
            perm = torch.randperm(n, device=dev)
            units += [(W, perm[i:i + bs]) for i in range(0, n, bs)]
        for j in torch.randperm(len(units)).tolist():
            W, sl = units[j]
            p = train_packs[W]
            opt.zero_grad()
            logit = _forward(model, arm, p, sl)
            loss = lossf(logit, p["y"][sl])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for W in EVAL_WS:
            p = test_packs[W]
            pred = (_forward(model, arm, p) > 0).float()
            out[W] = float((pred == p["y"]).float().mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"Stage C.0 gate  train={TRAIN_WS} eval={EVAL_WS} seeds={SEEDS} arms={ARMS}")
    curves = {}
    for arm in ARMS:
        mean = {W: float(np.mean([train_arm(arm, s)[W] for s in SEEDS])) for W in EVAL_WS}
        curves[arm] = mean
        print(f"  [{arm:7s}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in EVAL_WS))

    bigW = EVAL_WS[-1]
    learns = curves["layout"][8] >= 0.75
    beats_base = curves["layout"][bigW] - max(curves["settf"][bigW], curves["knn"][bigW]) >= 0.10
    beats_rand = curves["layout"][bigW] - curves["random"][bigW] >= 0.10
    verdict = {
        "layout_at_8": curves["layout"][8],
        "layout_minus_bestbase_at_bigW": curves["layout"][bigW] - max(curves["settf"][bigW], curves["knn"][bigW]),
        "layout_minus_random_at_bigW": curves["layout"][bigW] - curves["random"][bigW],
        "learns": bool(learns), "beats_baselines": bool(beats_base),
        "beats_random": bool(beats_rand),
        "PASS": bool(learns and beats_base and beats_rand),
    }
    out = {"config": {"train_ws": TRAIN_WS, "eval_ws": EVAL_WS, "seeds": SEEDS,
                      "hidden": HIDDEN, "epochs": EPOCHS, "n_train": N_TRAIN},
           "curves": curves, "verdict": verdict, "runtime_sec": round(time.time() - t0, 1)}
    from pathlib import Path
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase13_learned_layout"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "gate.json", "w") as f:
        json.dump(out, f, indent=2)
    v = verdict
    print(f"\nVERDICT: PASS={v['PASS']}  (learns@8={v['layout_at_8']:.2f}≥0.75={v['learns']}; "
          f"layout−bestbase@{bigW}={v['layout_minus_bestbase_at_bigW']:+.2f}≥0.10={v['beats_baselines']}; "
          f"layout−random@{bigW}={v['layout_minus_random_at_bigW']:+.2f}≥0.10={v['beats_random']})  "
          f"runtime={out['runtime_sec']}s")
    print(f"wrote {rep / 'gate.json'}")


if __name__ == "__main__":
    main()
