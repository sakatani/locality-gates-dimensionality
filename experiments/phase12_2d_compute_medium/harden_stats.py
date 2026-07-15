"""Phase 12 — statistical hardening for the Plan B manuscript (PAPER.md §9).

Re-runs the paper's load-bearing comparisons with 5 seeds and per-seed records,
then computes one-sided paired t-tests (H1: 2D-local > 1D-local) with Holm
correction across the headline tests. Thresholds/criteria are UNCHANGED from the
original pre-registrations — this is a confirmation with expanded seeds, and any
gate that flips under 5 seeds is reported as such.

Part 1 (§4, fixed-size W*): F1 with arms {2d, 1d, 1d_hilbert}; F3 control {2d, 1d}.
Part 2 (§5, length-gen suite): families {F1, F4b, F5} × arms {2d, 1d, gnn}.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats as sps

from . import stageb3
from .calibrate import REPORT_DIR, train_eval
from .probe import DEPTH, EPOCHS, HIDDEN, SWEEP, TAU, _dataset, _w_star

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
GAP_WS_FIXED = [w for w in (6, 8, 10, 12) if w in SWEEP]      # §4 headline range
GAP_WS_SUITE = [w for w in (8, 10, 12) if w in stageb3.TEST_WS]  # §5 pre-reg range


# ---------------------------------------------------------------------------
# Part 1 — fixed-size threshold sweep, per-seed
# ---------------------------------------------------------------------------

def sweep_per_seed(arm: str, family: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {W: [] for W in SWEEP}
    for s in SEEDS:
        for W in SWEEP:
            tr, va = _dataset(family, W, s)
            out[W].append(train_eval(arm, tr, va, hidden=HIDDEN, layers=DEPTH,
                                     epochs=EPOCHS, seed=s))
    mean = {W: float(np.mean(v)) for W, v in out.items()}
    per_seed_wstar = [_w_star([{"W": W, "acc": out[W][i]} for W in SWEEP])
                      for i in range(len(SEEDS))]
    print(f"  [{family}/{arm}] " + " ".join(f"{W}:{mean[W]:.2f}" for W in SWEEP)
          + f"  W*(mean-curve)={_w_star([{'W': W, 'acc': mean[W]} for W in SWEEP])}"
          + f"  W*(per-seed)={per_seed_wstar}")
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def per_seed_gap(a: dict[int, list[float]], b: dict[int, list[float]],
                 ws: list[int]) -> list[float]:
    """Per-seed mean over ``ws`` of (a − b)."""
    return [float(np.mean([a[W][i] - b[W][i] for W in ws]))
            for i in range(len(SEEDS))]


def one_sided_paired_t(gaps: list[float]) -> dict:
    g = np.asarray(gaps, dtype=np.float64)
    t, p_two = sps.ttest_1samp(g, 0.0)
    p = float(p_two / 2 if t > 0 else 1 - p_two / 2)
    return {"gaps_per_seed": [round(x, 4) for x in gaps],
            "mean": float(g.mean()), "sd": float(g.std(ddof=1)),
            "t": float(t), "p_one_sided": p, "n_seeds": len(gaps)}


def holm(tests: dict[str, dict]) -> None:
    """Adds holm-adjusted p in place across the given tests."""
    items = sorted(tests.items(), key=lambda kv: kv[1]["p_one_sided"])
    m = len(items)
    running = 0.0
    for rank, (name, res) in enumerate(items):
        adj = min(1.0, (m - rank) * res["p_one_sided"])
        running = max(running, adj)                # enforce monotonicity
        res["p_holm"] = running
        res["significant_at_0.05_holm"] = bool(running < 0.05)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print(f"harden_stats  seeds={SEEDS} sweep={SWEEP} suite_ws={stageb3.TEST_WS} quick={QUICK}")

    print("[part 1] fixed-size sweeps (F1 arms 2d/1d/1d_hilbert; F3 control):")
    f1 = {arm: sweep_per_seed(arm, "F1") for arm in ("2d", "1d", "1d_hilbert")}
    f3 = {arm: sweep_per_seed(arm, "F3") for arm in ("2d", "1d")}

    print("[part 2] length-gen suite per-seed (families F1/F4b/F5, arms 2d/1d/gnn):")
    suite: dict = {}
    for fam in ("F1", "F4b", "F5"):
        suite[fam] = {}
        for arm in ("2d", "1d", "gnn"):
            per_w: dict[int, list[float]] = {W: [] for W in stageb3.TEST_WS}
            for s in SEEDS:
                res = stageb3.train_suite(arm, fam, s)
                for W in stageb3.TEST_WS:
                    per_w[W].append(res[W])
            suite[fam][arm] = per_w
            mean = {W: float(np.mean(v)) for W, v in per_w.items()}
            print(f"  [{fam}/{arm}] " + " ".join(f"{W}:{mean[W]:.2f}"
                                                 for W in stageb3.TEST_WS))

    # Headline tests (one-sided paired t, H1: 2d > baseline), Holm-corrected.
    tests = {
        "fixed_F1_2d_vs_rowmajor": one_sided_paired_t(
            per_seed_gap(f1["2d"], f1["1d"], GAP_WS_FIXED)),
        "fixed_F1_2d_vs_hilbert": one_sided_paired_t(
            per_seed_gap(f1["2d"], f1["1d_hilbert"], GAP_WS_FIXED)),
        "suite_F1_2d_vs_1d": one_sided_paired_t(
            per_seed_gap(suite["F1"]["2d"], suite["F1"]["1d"], GAP_WS_SUITE)),
        "suite_F4b_2d_vs_1d": one_sided_paired_t(
            per_seed_gap(suite["F4b"]["2d"], suite["F4b"]["1d"], GAP_WS_SUITE)),
        "suite_F5_2d_vs_1d": one_sided_paired_t(
            per_seed_gap(suite["F5"]["2d"], suite["F5"]["1d"], GAP_WS_SUITE)),
    }
    holm(tests)

    # F3 control (expect NO 2d advantage): report two-sided, no Holm membership.
    f3_gaps = per_seed_gap(f3["2d"], f3["1d"], SWEEP)
    g = np.asarray(f3_gaps)
    t3, p3 = sps.ttest_1samp(g, 0.0)
    control = {"gaps_per_seed": [round(x, 4) for x in f3_gaps],
               "mean": float(g.mean()), "sd": float(g.std(ddof=1)),
               "t": float(t3), "p_two_sided": float(p3)}

    # Pre-registered gate re-checks on 5-seed mean curves (criteria unchanged).
    def mean_curve(d): return {W: float(np.mean(v)) for W, v in d.items()}
    def wstar(d): return _w_star([{"W": W, "acc": a} for W, a in mean_curve(d).items()])
    gate_checks = {
        "fixed_wstar_2d": wstar(f1["2d"]), "fixed_wstar_1d": wstar(f1["1d"]),
        "fixed_wstar_hilbert": wstar(f1["1d_hilbert"]),
        "fixed_gap_ge2_rowmajor": bool(wstar(f1["2d"]) - wstar(f1["1d"]) >= 2),
        "fixed_gap_ge2_hilbert": bool(wstar(f1["2d"]) - wstar(f1["1d_hilbert"]) >= 2),
    }
    for fam in ("F1", "F4b", "F5"):
        m2, m1 = mean_curve(suite[fam]["2d"]), mean_curve(suite[fam]["1d"])
        mg = mean_curve(suite[fam]["gnn"])
        gate_checks[f"suite_{fam}"] = {
            "mean_gap_le12": float(np.mean([m2[w] - m1[w] for w in GAP_WS_SUITE])),
            "two_ge_one_everywhere": bool(all(m2[w] >= m1[w] for w in stageb3.TEST_WS)),
            "gnn_control_at8": mg.get(8, 0.0),
            "PASS": bool(np.mean([m2[w] - m1[w] for w in GAP_WS_SUITE]) >= 0.10
                         and all(m2[w] >= m1[w] for w in stageb3.TEST_WS)
                         and mg.get(8, 0.0) >= 0.70),
        }

    out = {
        "config": {"seeds": SEEDS, "sweep": SWEEP, "suite_ws": stageb3.TEST_WS,
                   "tau": TAU, "gap_ws_fixed": GAP_WS_FIXED,
                   "gap_ws_suite": GAP_WS_SUITE, "quick": QUICK},
        "fixed_f1": {arm: {str(W): v for W, v in d.items()} for arm, d in f1.items()},
        "fixed_f3": {arm: {str(W): v for W, v in d.items()} for arm, d in f3.items()},
        "suite": {fam: {arm: {str(W): v for W, v in d.items()}
                        for arm, d in fams.items()} for fam, fams in suite.items()},
        "tests_holm": tests, "f3_control": control, "gate_checks": gate_checks,
        "runtime_sec": round(time.time() - t0, 1),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "harden_stats.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nTESTS (one-sided paired t, Holm-corrected):")
    for name, res in tests.items():
        print(f"  {name}: mean={res['mean']:+.3f}±{res['sd']:.3f} t={res['t']:.2f} "
              f"p={res['p_one_sided']:.2e} p_holm={res['p_holm']:.2e} "
              f"sig={res['significant_at_0.05_holm']}")
    print(f"F3 control: mean={control['mean']:+.3f} p_two={control['p_two_sided']:.3f} "
          f"(expect non-advantage)")
    print(f"GATES: {json.dumps({k: v for k, v in gate_checks.items() if not isinstance(v, dict)})}")
    for fam in ("F1", "F4b", "F5"):
        print(f"  suite_{fam}: {gate_checks[f'suite_{fam}']}")
    print(f"runtime={out['runtime_sec']}s\nwrote {REPORT_DIR / 'harden_stats.json'}")


if __name__ == "__main__":
    main()
