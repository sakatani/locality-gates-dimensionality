"""Phase 12 — Stage A.0.7: capable iterative-global baseline (PLAN; thresholds in
``STAGE_A07_PREREG.md``). A properly-built Universal Transformer (`univ`,
pre-LN + step embedding + grad-clip) vs 2D-local — settles "2D or iteration?".
"""

from __future__ import annotations

import json
import time

import numpy as np

from .calibrate import REPORT_DIR, train_eval
from .families import build_f1
from .models import Local2D, UniversalTransformer
from .probe import DEPTH, EPOCHS, HIDDEN, N_TRAIN, N_VAL, SWEEP, TAU, WALL_P, _w_star


# Give the iterative-global baseline a FAIR, STRONG shot: param-parity with 2D
# (hidden 96 ≈ 120k params vs 2D's 149k) + extra epochs + grad-clip + lr 1e-3.
# A loss under these conditions is robust; a win is real. (pre-reg: "capable".)
UNIV_HIDDEN = 96
UNIV_EPOCHS = max(EPOCHS, 32)


def _acc(arm: str, W: int, seeds: list[int], *, hidden: int, epochs: int,
         lr: float, clip) -> float:
    accs = []
    for s in seeds:
        tr = build_f1(N_TRAIN, W, W, WALL_P, 100 + s * 17 + W)
        va = build_f1(N_VAL, W, W, WALL_P, 900 + s * 17 + W)
        accs.append(train_eval(arm, tr, va, hidden=hidden, layers=DEPTH,
                               epochs=epochs, seed=s, lr=lr, clip=clip))
    return float(np.mean(accs))


def main() -> None:
    t0 = time.time()
    print(f"Stage A.0.7 capable iterative-global  sweep={SWEEP} iters={DEPTH} tau={TAU}")
    out: dict = {"config": {"sweep": SWEEP, "iters": DEPTH, "hidden_2d": HIDDEN,
                            "hidden_univ": UNIV_HIDDEN, "epochs_univ": UNIV_EPOCHS,
                            "tau": TAU, "epochs": EPOCHS, "n_train": N_TRAIN}}

    univ_seeds, td_seeds = [0, 1], [0, 1, 2]
    print(f"[univ] Universal Transformer (pre-LN + step-emb + clip, lr=1e-3, "
          f"hidden={UNIV_HIDDEN}, epochs={UNIV_EPOCHS}):")
    univ = [{"W": W, "acc": _acc("univ", W, univ_seeds, hidden=UNIV_HIDDEN,
                                 epochs=UNIV_EPOCHS, lr=1e-3, clip=1.0)} for W in SWEEP]
    print("  " + " ".join(f"{c['W']}:{c['acc']:.2f}" for c in univ) +
          f"  -> W*={_w_star(univ)}")
    print("[2d] 2D-local (reference, same regime):")
    td = [{"W": W, "acc": _acc("2d", W, td_seeds, hidden=HIDDEN, epochs=EPOCHS,
                               lr=2e-3, clip=None)} for W in SWEEP]
    print("  " + " ".join(f"{c['W']}:{c['acc']:.2f}" for c in td) +
          f"  -> W*={_w_star(td)}")

    p_univ = sum(p.numel() for p in UniversalTransformer(UNIV_HIDDEN, iters=DEPTH).parameters())
    p_2d = sum(p.numel() for p in Local2D(HIDDEN, DEPTH).parameters())
    w_univ, w_2d = _w_star(univ), _w_star(td)
    valid = w_univ >= 4
    if not valid:
        framing = "INCONCLUSIVE: univ failed to train (W*<4); fall back on GNN as the iterative reference"
    elif w_2d > w_univ:
        framing = "2D-local BEATS a capable iterative-global on the task itself (W*_2d > W*_univ)"
    else:
        framing = "iterated global MATCHES/EXCEEDS 2D on accuracy; 2D's case rests on compute (O(N*k) vs O(N^2)/step)"

    out.update({
        "univ_curve": univ, "td_curve": td,
        "w_star_univ": w_univ, "w_star_2d": w_2d,
        "params_univ": p_univ, "params_2d": p_2d,
        "per_step_cost": {"univ": "O(N^2) attention", "2d": "O(N*9) conv"},
        "baseline_valid": bool(valid), "framing": framing,
        "runtime_sec": round(time.time() - t0, 1),
    })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "a07.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFRAMING: {framing}")
    print(f"  W*_univ={w_univ} ({p_univ} params, O(N^2)/step) vs "
          f"W*_2d={w_2d} ({p_2d} params, O(N*9)/step)  runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'a07.json'}")


if __name__ == "__main__":
    main()
