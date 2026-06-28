"""Phase 12 — Stage A.0.5 decisive probe (PLAN §6/§7; thresholds in
``runs/phase12_2d_compute_medium/STAGE_A05_PREREG.md``, fixed before this runs).

Headline gate: threshold-crossing W* on F1 — largest W (square) with mean-over-
seeds acc ≥ τ=0.80 at matched depth-16 budget. PASS iff W*_2D − W*_1D ≥ 2.
Control: same on F3, REQUIRE W*_2D − W*_1D ≤ 1. Supporting: depth∝W monotone gap.
Diagnostics: 2D-vs-global efficiency (params), GNN W*.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .calibrate import REPORT_DIR, train_eval
from .families import build_f1, build_f3
from .models import GlobalAttn, Local1D, Local2D

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TAU = 0.80
SWEEP = [4, 6, 8] if QUICK else [4, 6, 8, 10, 12]
SEEDS = [0, 1] if QUICK else [0, 1, 2]
DEPTH = 16
HIDDEN = 32
# Full-run training budget = the validated Stage A.0 regime (so the task is
# solvable across the sweep; training budget was NOT a pre-registered variable —
# only τ/sweep/seeds/depth/wall_p were, see STAGE_A05_PREREG.md).
N_TRAIN = 2000 if QUICK else 4000
N_VAL = 800 if QUICK else 1200
EPOCHS = 12 if QUICK else 22
WALL_P = 0.28


def _dataset(family: str, W: int, seed: int):
    if family == "F1":
        tr = build_f1(N_TRAIN, W, W, WALL_P, 100 + seed * 17 + W)
        va = build_f1(N_VAL, W, W, WALL_P, 900 + seed * 17 + W)
    else:
        tr = build_f3(N_TRAIN, W, W, 100 + seed * 17 + W)
        va = build_f3(N_VAL, W, W, 900 + seed * 17 + W)
    return tr, va


def _mean_acc(arm: str, family: str, W: int, depth: int, seeds: list[int]) -> float:
    accs = []
    for s in seeds:
        tr, va = _dataset(family, W, s)
        accs.append(train_eval(arm, tr, va, hidden=HIDDEN, layers=depth,
                               epochs=EPOCHS, seed=s))
    return float(np.mean(accs))


def _w_star(curve: list[dict]) -> int:
    """Largest W with mean acc ≥ TAU (0 if none); per pre-reg, literal max."""
    clear = [c["W"] for c in curve if c["acc"] >= TAU]
    return max(clear) if clear else 0


def threshold_sweep(family: str) -> dict:
    out = {"family": family, "tau": TAU, "depth": DEPTH, "seeds": SEEDS}
    for arm in ("2d", "1d"):
        curve = [{"W": W, "acc": _mean_acc(arm, family, W, DEPTH, SEEDS)}
                 for W in SWEEP]
        out[arm] = {"curve": curve, "w_star": _w_star(curve)}
        pretty = " ".join(f"{c['W']}:{c['acc']:.2f}" for c in curve)
        print(f"  [{family}/{arm}] {pretty}  -> W*={out[arm]['w_star']}")
    out["w_star_gap"] = out["2d"]["w_star"] - out["1d"]["w_star"]
    return out


def depth_scaled() -> list[dict]:
    seeds = SEEDS[:2]
    rows = []
    for W in SWEEP:
        depth = 2 * W
        a2 = _mean_acc("2d", "F1", W, depth, seeds)
        a1 = _mean_acc("1d", "F1", W, depth, seeds)
        rows.append({"W": W, "depth": depth, "acc_2d": a2, "acc_1d": a1,
                     "gap": a2 - a1})
        print(f"  [depth=2W] W={W} depth={depth}: 2D={a2:.2f} 1D={a1:.2f} gap={a2-a1:+.2f}")
    return rows


def efficiency() -> dict:
    W = 6 if QUICK else 8
    tr, va = _dataset("F1", W, 0)
    a2 = train_eval("2d", tr, va, hidden=HIDDEN, layers=DEPTH, epochs=EPOCHS, seed=0)
    ag = train_eval("global", tr, va, hidden=HIDDEN, layers=DEPTH, epochs=EPOCHS, seed=0)
    p2 = sum(p.numel() for p in Local2D(HIDDEN, DEPTH).parameters())
    pg = sum(p.numel() for p in GlobalAttn(HIDDEN, layers=min(DEPTH, 4)).parameters())
    print(f"  [eff W={W}] 2D acc={a2:.2f} ({p2} params) vs global acc={ag:.2f} ({pg} params)")
    return {"W": W, "acc_2d": a2, "params_2d": p2, "acc_global": ag, "params_global": pg}


def gnn_diag() -> dict:
    curve = [{"W": W, "acc": _mean_acc("gnn", "F1", W, DEPTH, SEEDS[:1])} for W in SWEEP]
    pretty = " ".join(f"{c['W']}:{c['acc']:.2f}" for c in curve)
    print(f"  [gnn] {pretty}  -> W*={_w_star(curve)}")
    return {"curve": curve, "w_star": _w_star(curve)}


def main() -> None:
    t0 = time.time()
    print(f"Stage A.0.5 probe  sweep={SWEEP} seeds={SEEDS} tau={TAU} quick={QUICK}")
    out: dict = {"config": {"sweep": SWEEP, "seeds": SEEDS, "tau": TAU,
                            "depth": DEPTH, "hidden": HIDDEN, "n_train": N_TRAIN,
                            "epochs": EPOCHS, "wall_p": WALL_P, "quick": QUICK}}
    print("[headline] F1 threshold-crossing W*:")
    out["f1"] = threshold_sweep("F1")
    print("[control] F3 threshold-crossing W*:")
    out["f3"] = threshold_sweep("F3")
    print("[supporting] depth ∝ W:")
    out["depth_scaled"] = depth_scaled()
    print("[diagnostic] efficiency 2D vs global:")
    out["efficiency"] = efficiency()
    print("[diagnostic] GNN W*:")
    out["gnn"] = gnn_diag()

    f1_pass = out["f1"]["w_star_gap"] >= 2
    f3_ok = out["f3"]["w_star_gap"] <= 1
    deltas = [r["gap"] for r in out["depth_scaled"]]
    gap_monotone = all(b >= a - 0.03 for a, b in zip(deltas, deltas[1:]))
    out["verdict"] = {
        "f1_w_star_gap": out["f1"]["w_star_gap"],
        "f3_w_star_gap": out["f3"]["w_star_gap"],
        "f1_headline_pass": bool(f1_pass),
        "f3_control_ok": bool(f3_ok),
        "depth_scaled_gap_monotone": bool(gap_monotone),
        "PASS": bool(f1_pass and f3_ok),
    }
    out["runtime_sec"] = round(time.time() - t0, 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "probe.json", "w") as f:
        json.dump(out, f, indent=2)
    v = out["verdict"]
    print(f"\nVERDICT: PASS={v['PASS']}  "
          f"(F1 W*-gap={v['f1_w_star_gap']} need≥2; F3 W*-gap={v['f3_w_star_gap']} need≤1; "
          f"depth∝W monotone={v['depth_scaled_gap_monotone']})  runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'probe.json'}")


if __name__ == "__main__":
    main()
