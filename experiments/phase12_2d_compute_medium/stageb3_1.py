"""Phase 12 — Stage B.3.1: F4 fixed-K redesign (PLAN; thresholds in
``STAGE_B31_PREREG.md``). Re-run the metric family with the scale-invariant
bounded-radius fixed-K builder (F4b) under the identical B.1/B.3 gate.
"""

from __future__ import annotations

import json
import time

from .calibrate import REPORT_DIR
from .stageb3 import ARMS, _gate, _mean_curve


def main() -> None:
    t0 = time.time()
    print("Stage B.3.1  F4b (bounded-radius R=8, fixed K=6, scale-invariant)")
    curves = {arm: _mean_curve(arm, "F4b") for arm in ARMS}
    for arm in ARMS:
        print(f"  [{arm}] " + " ".join(f"{W}:{curves[arm][W]:.2f}" for W in curves[arm]))
    gate = _gate(curves)
    out = {"family": "F4b", "curves": curves, "gate": gate,
           "runtime_sec": round(time.time() - t0, 1)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "stageb3_1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: F4b PASS={gate['PASS']} (gap≤12={gate['mean_gap_le12']:+.2f} "
          f"2D≥1D={gate['two_ge_one_everywhere']} gnn@8={gate['gnn_control_at8']:.2f})  "
          f"runtime={out['runtime_sec']}s")
    print(f"wrote {REPORT_DIR / 'stageb3_1.json'}")


if __name__ == "__main__":
    main()
