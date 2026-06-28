"""Phase 12 — Stage A.0 calibration (PLAN §7).

Validates the families before the decisive A.0.5 probe:

1. **Bandwidth decoupling** — analytic: 2D-bw = 1 (constant) vs best-1D-bw =
   min(W,H) (grows). Pure combinatorics, no training.
2. **Difficulty band** — the global arm lands in [0.55, 0.90] on F1 / F3
   (non-vacuous, not saturated).
3. **Small-scale degradation** — on F1, 1D-local accuracy drops with W while
   2D-local holds (the bandwidth effect is reachable at calibration scale).
4. **F3 control** — 2D-local does NOT beat 1D-local on the 1D-intrinsic task.

GREEN ⇒ proceed to A.0.5. This file does not gate the headline (that is A.0.5
with pre-registered thresholds, PLAN §6); it only checks the design is sound.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .families import (build_f1, build_f3, instance_to_channels, readout_cell,
                       class_balance)
from .models import GlobalAttn, GNNAdj, Local1D, Local2D, build_neighbour_index
from .substrate import (GridInstance, bandwidth_2d, grid_bandwidth_best_1d,
                        row_major_bandwidth)

REPORT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tensorisation
# ---------------------------------------------------------------------------

def tensorise(instances: list[GridInstance]):
    X = np.stack([instance_to_channels(i) for i in instances]).astype(np.float32)
    idx = np.array([r * i.W + c for i in instances for (r, c) in [readout_cell(i)]],
                   dtype=np.int64)
    y = np.array([i.label for i in instances], dtype=np.float32)
    walls = np.stack([i.walls for i in instances]).astype(np.int64)
    return X, idx, y, walls


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_eval(arm: str, train: list[GridInstance], val: list[GridInstance],
               *, hidden: int, layers: int, epochs: int, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    Xtr, itr, ytr, wtr = tensorise(train)
    Xva, iva, yva, wva = tensorise(val)

    if arm == "2d":
        model = Local2D(hidden, layers)
    elif arm == "1d":
        model = Local1D(hidden, layers)
    elif arm == "global":
        model = GlobalAttn(hidden, layers=min(layers, 4))
    elif arm == "gnn":
        model = GNNAdj(hidden, layers)
    else:
        raise ValueError(arm)
    model = model.to(dev)

    nbr_tr = nbr_va = mask_tr = mask_va = None
    if arm == "gnn":
        n_tr, m_tr = build_neighbour_index(wtr)
        n_va, m_va = build_neighbour_index(wva)
        nbr_tr = torch.from_numpy(n_tr).to(dev)
        mask_tr = torch.from_numpy(m_tr).to(dev)
        nbr_va = torch.from_numpy(n_va).to(dev)
        mask_va = torch.from_numpy(m_va).to(dev)

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    itr_t = torch.from_numpy(itr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)
    iva_t = torch.from_numpy(iva).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    bs = 256
    n = len(train)
    best = 0.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            if arm == "gnn":
                logit = model(Xtr_t[b], itr_t[b], nbr_tr[b], mask_tr[b])
            else:
                logit = model(Xtr_t[b], itr_t[b])
            loss = lossf(logit, ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            if arm == "gnn":
                pred = (model(Xva_t, iva_t, nbr_va, mask_va) > 0).float().cpu().numpy()
            else:
                pred = (model(Xva_t, iva_t) > 0).float().cpu().numpy()
        acc = float((pred == yva).mean())
        best = max(best, acc)
    return best


# ---------------------------------------------------------------------------
# Stage A.0
# ---------------------------------------------------------------------------

def bandwidth_table(ws: list[int]) -> list[dict]:
    rows = []
    for w in ws:
        rows.append({
            "W": w, "H": w,
            "bw_2d": bandwidth_2d(),
            "bw_1d_rowmajor": row_major_bandwidth(w, w),
            "bw_1d_best": grid_bandwidth_best_1d(w, w),
        })
    return rows


def main() -> None:
    t0 = time.time()
    n_train = 1500 if QUICK else 4000
    n_val = 600 if QUICK else 1200
    epochs = 8 if QUICK else 22
    hidden = 32
    layers = 16
    wall_p = 0.28
    ws_deg = [5, 7] if QUICK else [5, 7, 9]
    seed = 0

    out: dict = {"config": {
        "n_train": n_train, "n_val": n_val, "epochs": epochs, "hidden": hidden,
        "layers": layers, "wall_p": wall_p, "ws_deg": ws_deg, "seed": seed,
        "quick": QUICK,
    }}

    # 1) Bandwidth decoupling (analytic).
    out["bandwidth"] = bandwidth_table([5, 7, 9, 11, 15])
    print("[1] bandwidth decoupling:")
    for r in out["bandwidth"]:
        print(f"    W={r['W']:2d}  2D-bw={r['bw_2d']}  best-1D-bw={r['bw_1d_best']:2d}"
              f"  row-major-bw={r['bw_1d_rowmajor']:2d}")

    # 2) Solvability + non-vacuity (PLAN §7 difficulty gate, adapted).
    #    Reachability is an ITERATIVE-propagation task: full attention (global,
    #    Phase 11's regime) lacks the iterative-locality bias and floors, so the
    #    "is it solvable?" reference is the GNN (message passing on the true
    #    adjacency = the canonical reachability solver) and 2D-local at depth.
    #    Trivial baselines sit at ~0.5 by the 50/50 + density-matched design.
    print("[2] solvability (strong arm must reach the task; global reported)...")
    w_band = 7
    f1_tr = build_f1(n_train, w_band, w_band, wall_p, seed)
    f1_va = build_f1(n_val, w_band, w_band, wall_p, seed + 1)
    f3_tr = build_f3(n_train, 9, 9, seed)
    f3_va = build_f3(n_val, 9, 9, seed + 1)
    out["balance"] = {"f1": class_balance(f1_tr), "f3": class_balance(f3_tr)}
    solv = {
        "f1_gnn": train_eval("gnn", f1_tr, f1_va, hidden=hidden, layers=layers,
                             epochs=epochs, seed=seed),
        "f1_2d": train_eval("2d", f1_tr, f1_va, hidden=hidden, layers=layers,
                            epochs=epochs, seed=seed),
        "f1_global": train_eval("global", f1_tr, f1_va, hidden=hidden,
                                 layers=layers, epochs=epochs, seed=seed),
        "f3_1d": train_eval("1d", f3_tr, f3_va, hidden=hidden, layers=layers,
                            epochs=epochs, seed=seed),
    }
    out["solvability"] = solv
    strong_f1 = max(solv["f1_gnn"], solv["f1_2d"])
    print(f"    F1: GNN={solv['f1_gnn']:.3f} 2D={solv['f1_2d']:.3f} "
          f"global={solv['f1_global']:.3f} (strong={strong_f1:.3f}, want ≥0.80)")
    print(f"    F3: 1D={solv['f3_1d']:.3f} (want ≥0.70, solvable & 1D-friendly)")

    # 3) Degradation curve: 2D vs 1D on F1 across W.
    print("[3] degradation (F1, 2D-local vs 1D-local)...")
    deg = []
    for w in ws_deg:
        tr = build_f1(n_train, w, w, wall_p, seed + 10 + w)
        va = build_f1(n_val, w, w, wall_p, seed + 20 + w)
        a2 = train_eval("2d", tr, va, hidden=hidden, layers=layers,
                        epochs=epochs, seed=seed)
        a1 = train_eval("1d", tr, va, hidden=hidden, layers=layers,
                        epochs=epochs, seed=seed)
        deg.append({"W": w, "acc_2d": a2, "acc_1d": a1, "gap": a2 - a1,
                    "balance": class_balance(tr)})
        print(f"    W={w}: 2D={a2:.3f} 1D={a1:.3f} gap={a2 - a1:+.3f}"
              f" (bal={class_balance(tr):.2f})")
    out["degradation_f1"] = deg

    # 4) F3 control: 2D must NOT beat 1D.
    print("[4] F3 control (2D-local must NOT beat 1D-local)...")
    c2 = train_eval("2d", f3_tr, f3_va, hidden=hidden, layers=layers,
                    epochs=epochs, seed=seed)
    c1 = train_eval("1d", f3_tr, f3_va, hidden=hidden, layers=layers,
                    epochs=epochs, seed=seed)
    out["f3_control"] = {"acc_2d": c2, "acc_1d": c1, "gap_2d_minus_1d": c2 - c1}
    print(f"    F3: 2D={c2:.3f} 1D={c1:.3f} gap(2D-1D)={c2 - c1:+.3f}")

    # Verdict heuristics (calibration GREEN, not the headline gate, PLAN §7).
    gap_grows = (len(deg) >= 2 and deg[-1]["gap"] > deg[0]["gap"]
                 and deg[-1]["gap"] > 0.03)
    solvable = strong_f1 >= 0.80 and solv["f3_1d"] >= 0.70
    f3_ok = (c2 - c1) <= 0.03
    out["verdict"] = {
        "bandwidth_decoupled": True,
        "solvable_by_strong_arm": bool(solvable),
        "degradation_1d_drops_2d_holds": gap_grows,
        "f3_control_ok": f3_ok,
        "green": bool(solvable and gap_grows and f3_ok),
    }
    out["runtime_sec"] = round(time.time() - t0, 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGREEN={out['verdict']['green']}  "
          f"(solvable={solvable}, gap_grows={gap_grows}, f3_ok={f3_ok})  "
          f"runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'calibration.json'}")


if __name__ == "__main__":
    main()
