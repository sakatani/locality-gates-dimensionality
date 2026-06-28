"""Phase 12 — Stage A.0.6 adversarial hardening (PLAN; thresholds in
``STAGE_A06_PREREG.md``, fixed before this runs).

Threat 1 (GATING): Hilbert/space-filling 1D control — does the best 1D
serialisation rescue 1D? PASS iff W*_2D − W*_1d_hilbert ≥ 2.
Threat 2 (reported): capable looped-global — is the advantage 2D or iteration?
"""

from __future__ import annotations

import json
import time

from .calibrate import REPORT_DIR
from .probe import (DEPTH, EPOCHS, HIDDEN, SEEDS, SWEEP, TAU, _mean_acc,
                    _w_star)


def sweep_arm(arm: str, seeds: list[int]) -> dict:
    curve = [{"W": W, "acc": _mean_acc(arm, "F1", W, DEPTH, seeds)} for W in SWEEP]
    pretty = " ".join(f"{c['W']}:{c['acc']:.2f}" for c in curve)
    print(f"  [{arm}] {pretty}  -> W*={_w_star(curve)}")
    return {"curve": curve, "w_star": _w_star(curve), "seeds": seeds}


def main() -> None:
    t0 = time.time()
    print(f"Stage A.0.6 hardening  sweep={SWEEP} seeds={SEEDS} tau={TAU}")
    out: dict = {"config": {"sweep": SWEEP, "seeds": SEEDS, "tau": TAU,
                            "depth": DEPTH, "hidden": HIDDEN, "epochs": EPOCHS}}

    print("[threat 1] Hilbert/space-filling 1D control (gating):")
    out["arm_2d"] = sweep_arm("2d", SEEDS)
    out["arm_1d_rowmajor"] = sweep_arm("1d", SEEDS)
    out["arm_1d_hilbert"] = sweep_arm("1d_hilbert", SEEDS)

    print("[threat 2] capable looped-global baseline (reported):")
    out["arm_looped"] = sweep_arm("looped", SEEDS[:2])

    w2 = out["arm_2d"]["w_star"]
    wh = out["arm_1d_hilbert"]["w_star"]
    wr = out["arm_1d_rowmajor"]["w_star"]
    wl = out["arm_looped"]["w_star"]
    hilbert_survives = (w2 - wh) >= 2
    out["verdict"] = {
        "w_star_2d": w2,
        "w_star_1d_rowmajor": wr,
        "w_star_1d_hilbert": wh,
        "w_star_looped": wl,
        "hilbert_gap_2d_minus_hilbert": w2 - wh,
        "hilbert_control_survives": bool(hilbert_survives),
        "looped_solves_like_2d": bool((w2 - wl) <= 1),
        "PASS": bool(hilbert_survives),
    }
    out["runtime_sec"] = round(time.time() - t0, 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "harden.json", "w") as f:
        json.dump(out, f, indent=2)
    v = out["verdict"]
    print(f"\nVERDICT: PASS={v['PASS']}  "
          f"(Hilbert: W*_2D={w2} vs W*_hilbert={wh}, gap={w2-wh} need≥2; "
          f"W*_rowmajor={wr}; looped W*={wl}, solves-like-2D={v['looped_solves_like_2d']})  "
          f"runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'harden.json'}")


if __name__ == "__main__":
    main()
