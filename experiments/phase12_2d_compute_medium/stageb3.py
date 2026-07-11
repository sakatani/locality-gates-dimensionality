"""Phase 12 — Stage B.3: reasoning-task suite length-generalization (PLAN;
thresholds in ``STAGE_B3_PREREG.md``). Is 2D-local length-gen a GENERAL property
across {connectivity F1, metric F4, dynamics F5}, or reachability-specific?

Reuses B.1's mixed-W trainer (`lengthgen._pack` / `_forward`), parameterised by
family builder.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .calibrate import REPORT_DIR, device
from .families import build_f1, build_f4, build_f5
from .lengthgen import _forward, _pack
from .models import GNNAdj, Local1D, Local2D

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TRAIN_WS = [4, 5, 6]
TEST_WS = [8, 12, 16] if QUICK else [8, 10, 12, 16, 20]
SEEDS = [0, 1] if QUICK else [0, 1, 2]
HIDDEN, DEPTH = 32, 16
N_TRAIN = 1500 if QUICK else 3000
N_TEST = 600 if QUICK else 1000
EPOCHS = 10 if QUICK else 22
WALL_P = 0.28
ARMS = ["2d", "1d", "gnn"]


def _build(family: str, n: int, W: int, seed: int):
    if family == "F1":
        return build_f1(n, W, W, WALL_P, seed)
    if family == "F4":
        return build_f4(n, W, W, WALL_P, seed)
    return build_f5(n, W, W, seed)


def train_suite(arm: str, family: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = {"2d": Local2D, "1d": Local1D, "gnn": GNNAdj}[arm](HIDDEN, DEPTH).to(dev)
    train_packs = {W: _pack(_build(family, N_TRAIN, W, 300 + seed * 11 + W), dev)
                   for W in TRAIN_WS}
    test_packs = {W: _pack(_build(family, N_TEST, W, 800 + seed * 11 + W), dev)
                  for W in TEST_WS}

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
            sub = {"X": p["X"][bidx], "idx": p["idx"][bidx],
                   "walls": p["walls"][bidx.cpu().numpy()]}
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


def _mean_curve(arm: str, family: str) -> dict:
    per_seed = [train_suite(arm, family, s) for s in SEEDS]
    return {W: float(np.mean([r[W] for r in per_seed])) for W in TEST_WS}


def _gate(curves: dict) -> dict:
    mean_ws = [w for w in (8, 10, 12) if w in TEST_WS]
    gap = float(np.mean([curves["2d"][w] - curves["1d"][w] for w in mean_ws]))
    two_ge_one = all(curves["2d"][w] >= curves["1d"][w] for w in TEST_WS)
    gnn_ctrl = curves["gnn"].get(8, 0.0) >= 0.70
    return {"mean_gap_le12": gap, "two_ge_one_everywhere": bool(two_ge_one),
            "gnn_control_at8": curves["gnn"].get(8, 0.0),
            "PASS": bool(gap >= 0.10 and two_ge_one and gnn_ctrl)}


def main() -> None:
    t0 = time.time()
    print(f"Stage B.3 suite  train={TRAIN_WS} test={TEST_WS} seeds={SEEDS} F1/F4/F5")
    out: dict = {"config": {"train_ws": TRAIN_WS, "test_ws": TEST_WS,
                            "seeds": SEEDS, "hidden": HIDDEN, "depth": DEPTH,
                            "epochs": EPOCHS}, "families": {}}
    for fam in ["F1", "F4", "F5"]:
        print(f"[{fam}]")
        curves = {arm: _mean_curve(arm, fam) for arm in ARMS}
        for arm in ARMS:
            print(f"  [{arm}] " + " ".join(f"{W}:{curves[arm][W]:.2f}" for W in TEST_WS))
        gate = _gate(curves)
        out["families"][fam] = {"curves": curves, "gate": gate}
        print(f"  gate: gap≤12={gate['mean_gap_le12']:+.2f} "
              f"2D≥1D={gate['two_ge_one_everywhere']} gnn@8={gate['gnn_control_at8']:.2f} "
              f"-> PASS={gate['PASS']}")

    new_pass = [f for f in ("F4", "F5") if out["families"][f]["gate"]["PASS"]]
    suite = ("PASS" if len(new_pass) == 2 else
             "PARTIAL" if len(new_pass) == 1 else "NULL")
    out["verdict"] = {"new_families_passed": new_pass, "suite": suite,
                      "f1_pass": out["families"]["F1"]["gate"]["PASS"]}
    out["runtime_sec"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "stageb3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: suite={suite} (new families passed: {new_pass or 'none'}; "
          f"F1={out['verdict']['f1_pass']})  runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'stageb3.json'}")


if __name__ == "__main__":
    main()
